import logging
import time

from flatland import Form, Boolean
from microdrop.plugin_helpers import AppDataController, StepOptionsController
from microdrop.plugin_manager import (IPlugin, Plugin, implements, emit_signal,
                                      get_service_instance_by_name,
                                      PluginGlobals, ScheduleRequest)

from microdrop.app_context import get_app
import numpy as np
import path_helpers as ph
import trollius as asyncio
import pandas as pd
from dropbot.chip import chip_info
from svg_model import svg_shapes_to_df

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

logger = logging.getLogger(__name__)

# Add plugin to `"microdrop.managed"` plugin namespace.
PluginGlobals.push_env('microdrop.managed')


class PlateauDetectionPlugin(AppDataController, StepOptionsController, Plugin):
    '''
    This class is automatically registered with the PluginManager.
    '''
    implements(IPlugin)

    plugin_name = str(ph.path(__file__).realpath().parent.name)
    try:
        version = __version__
    except NameError:
        version = 'v0.0.0+unknown'

    AppFields = None

    StepFields = Form.of(Boolean.named('Plateau Detection')
                         .using(default=False, optional=True),
                         Boolean.named('Check Split')
                         .using(default=False, optional=True),
                         Boolean.named('Calibrate Threshold')
                         .using(default=False, optional=True))

    def __init__(self):
        super(PlateauDetectionPlugin, self).__init__()
        # The `name` attribute is required in addition to the `plugin_name`
        # attribute because MicroDrop uses it for plugin labels in, for
        # example, the plugin manager dialog.
        self.name = self.plugin_name

        # `dropbot.SerialProxy` instance
        self.dropbot_remote = None

        # Latch to, e.g., config menus, only once
        self.initialized = False

        self.timeout = 15  # timeout for plateau detection

        self.active_step_kwargs = None

        #True if calibration for plateau detection has been done
        self.calibrated = False

        #Threshold for capacitance deviation per electrode for plateau detection
        self.stdev_threshold = None
        self.stdev_normalized = None

        #dataframe with the area of each electrode in the device
        self.electrode_areas = None


    def apply_step_options(self, step_options):
        '''
        Apply the specified step options.


        Parameters
        ----------
        step_options : dict
            Dictionary containing the MR-Box peripheral board plugin options
            for a protocol step.
        '''
        # app = get_app()
        # app_values = self.get_app_values()

        if step_options.get("Calibrate Threshold"):
            #step option to Calibrate the standard deviation for the dmf_device
            # Get actuated electrodes from electrode controller plugin

            electrode_list = self.get_activated_electrodes()
            self.turn_on_selected_electrodes(electrode_list=electrode_list)

            # calculate the total areas of actualted electrodes for the calibration
            actuated_area = 0
            for electrode in electrode_list:
                actuated_area = actuated_area + self.electrode_areas['area'][str(electrode)]
            logger.info(electrode_list)
            logger.info(actuated_area)
            #calculate the stdev several times for the same area
            stdev_n = []
            for i in range (0,20):
                # Measure capacitance in steps of 10
                x = []
                for i in range(0, 15):
                    cap = self.dropbot_remote.measure_capacitance()
                    x.append(cap)
                # Calculate standard deviation
                stdev = np.std(x)
                stdev_n.append(stdev)
            #Pick the highest stdev and normalize it by actuated area
            # stdev_top = max(stdev_n)
            #try the mean value instead
            stdev_mean = sum(stdev_n)/float(len(stdev_n))
            self.stdev_normalized = stdev_mean/actuated_area
            logger.info('Normalized standard deviation set at {} F/mm^2'.format(self.stdev_normalized))
            # logger.warning("Calibration complete")
            self.calibrated = True

        if step_options.get("Plateau Detection"):

            # Standard deviation initialized
            stdev = 1
            # Get actuated electrodes from electrode controller plugin
            electrode_list = self.get_activated_electrodes()
            self.turn_on_selected_electrodes(electrode_list=electrode_list)


            if self.calibrated:
                # calculate the total areas of actuated electrodes for threshold determination
                actuated_area = 0
                for electrode in electrode_list:
                    actuated_area = actuated_area + self.electrode_areas['area'][str(electrode)]

                self.stdev_threshold = actuated_area*self.stdev_normalized
                logger.info('Threshold set at {} F'.format(self.stdev_threshold))

                t0 = time.time()
                time.sleep(0.2)
            # If calibration hasn't been done use old thresholds
            else:
                logger.warning("You have not performed a capacitance calibration. Plateau detection will likely fail.")
                #Get number of actuated elecrodes
                num_actuated = len(electrode_list)
                # Threshold needs to change based on number of actuated electrodes
                if num_actuated>=7:
                    self.stdev_threshold = 7e-13 * num_actuated
                else:
                    self.stdev_threshold = 5e-12
                logger.info('Threshold set at {} F'.format(self.stdev_threshold))
                #stdev_threshold = 4e-12

            # Loop until plateau has been reached
            while stdev > self.stdev_threshold:
                # Measure capacitance in steps of 10
                x = []
                for i in range(0, 10):
                    cap = self.dropbot_remote.measure_capacitance()
                    x.append(cap)
                # Calculate standard deviation
                stdev = np.std(x)
                logger.info(stdev)

                if (time.time() - t0) > self.timeout:
                    logger.info('Timeout reached, process stopped')
                    break

        if step_options.get("Check Split"):
            #Get current voltage for later use
            self.current_voltage = self.active_step_kwargs[u'microdrop.electrode_controller_plugin'][
                u'Voltage (V)']
            # Get actuated electrodes from electrode controller plugin

            path_list = self.get_activated_paths()
            # Get electrode list to not include in electrodes to test
            activated_electrodes = self.get_activated_electrodes()

            for cycle in range(4):
                logger.info(cycle)
                # Turn voltage on ahead of testing so that the droplets won't get pulled together
                self.turn_on_selected_electrodes(voltage = 100, electrode_list=activated_electrodes)

                split_success = self.splitting_check(path_list, activated_electrodes)

                # If splitting failed wait a set time and test again untill success or timeout
                if not split_success:
                    if cycle==3:
                        logger.warning("Splitting Failed!")
                    self.current_voltage+=5
                    self.turn_on_selected_electrodes(voltage = self.current_voltage, electrode_list=activated_electrodes)
                    time.sleep(1.5)
                    #split_success = self.splitting_check(path_list, activated_electrodes)
                if split_success:
                    logger.info("Splitting succeeded")
                    break

    def splitting_check(self, path_list, activated_electrodes):

        # Electrode neighbours that are between two activated electrodes and not in electrode list
        electrodes_to_test = []
        # Electrodes neighbours that are not in electrode_list
        neighbour_list = []
        #Find the neighbour of the actuated electrodes
        for path in path_list:
            neighbours = self.identify_neighbours(path)
            # Iterate through all neighbors found, if it isn't in activated_electrodes
            # add it to neighbour_list, if it is already in neighbour_list add it to
            # electrodes_to_test (duplicates meaning that it's between two actuated electrodes)
            for neighbour in neighbours:
                if neighbour not in activated_electrodes:
                    if neighbour in neighbour_list:
                        electrodes_to_test.append(neighbour)
                    else:
                        neighbour_list.append(neighbour)
        #Remove duplicates from electrodes to test
        electrodes_to_test= list(set(electrodes_to_test))

        # Quickly actuate electrodes with a low voltage
        # self.turn_on_selected_electrodes(voltage=20, electrode_list=electrodes_to_test)
        # Measure capacitance
        #cap = self.dropbot_remote.measure_capacitance()
        # Use the same test as for find liquid
        cap = self.dropbot_remote.channel_capacitances(electrodes_to_test)
        logger.info("-"*72)
        logger.info("Capacitances: {} F".format(cap))
        logger.info("-"*72)
        #Turn off test electrodes and turn on electrodes from step
        self.turn_on_selected_electrodes(voltage = None, electrode_list=activated_electrodes)
        success_test = True
        for electrode in cap:
            if electrode > 7e-12:
                logger.info('Splitting failed on electrode {}!'.format(cap[cap == electrode].index[0]))
                success_test = False
        return success_test


    def identify_neighbours(self, electrode_id):
        '''
        .. versionadded:: 2.36

        Pulse each neighbour electrode to help visually identify an electrode.
        '''
        app = get_app()
        neighbours = app.dmf_device.electrode_neighbours.loc[electrode_id].dropna()
        neighbour_channels = \
            app.dmf_device.channels_by_electrode.loc[neighbours]
        #Remove duplicate neighbours
        neighbour_channels = list(set(neighbour_channels))
        # for channel in neighbour_channels:
        #     for state in (1, 0):
        #         self.control_board.state_of_channels = \
        #             pd.Series(state, index=[channel])
        return neighbour_channels

    def initialize_connection_with_dropbot(self):
        '''
        If the dropbot plugin is installed and enabled, try getting its
        reference.
        '''
        try:
            service = get_service_instance_by_name('dropbot_plugin')
            if service.enabled():
                self.dropbot_remote = service.control_board
            assert (self.dropbot_remote.properties.package_name == 'dropbot')
        except Exception:
            logger.debug('[%s] Could not communicate with Dropbot.', __name__,
                         exc_info=True)
            logger.warning('Could not communicate with DropBot.')


    def on_plugin_enable(self):
        '''
        Handler called when plugin is enabled.

        For example, when the MicroDrop application is **launched**, or when
        the plugin is **enabled** from the plugin manager dialog.
        '''

        try:
            super(PlateauDetectionPlugin, self).on_plugin_enable()
        except AttributeError:
            pass

    def on_plugin_disable(self):
        '''
        Handler called when plugin is disabled.

        For example, when the MicroDrop application is **closed**, or when the
        plugin is **disabled** from the plugin manager dialog.
        '''
        try:
            super(PlateauDetectionPlugin, self).on_plugin_disable()
        except AttributeError:
            pass

    @asyncio.coroutine
    def on_step_run(self, plugin_kwargs, signals):
        '''
        Handler called whenever a step is executed.

        Plugins that handle this signal **MUST** emit the ``on_step_complete``
        signal once they have completed the step.  The protocol controller will
        wait until all plugins have completed the current step before
        proceeding.
        '''
        # Get latest step field values for this plugin.
        self.active_step_kwargs = plugin_kwargs
        options = plugin_kwargs[self.name]

        if self.dropbot_remote is None:
            self.initialize_connection_with_dropbot()

        if self.electrode_areas is None:
            self.get_chip_info()

        # Apply step options
        self.apply_step_options(options)

        self.active_step_kwargs = None
        raise asyncio.Return()

    def on_protocol_run(self):
        '''
        Handler called when a protocol starts running.
        '''
        # TODO: this should be run in on_plugin_enable; however, the
        # mr-box-peripheral-board seems to have trouble connecting **after**
        # the DropBot has connected.
        self.initialize_connection_with_dropbot()
        self.get_chip_info()

    def get_chip_info(self):
        # Get the areas of the electrodes
        app = get_app()

        self.svg_path=ph.path(app.dmf_device.svg_filepath)
        chip_info_ = chip_info(self.svg_path)
        # self.path_to_id = chip_info_['electrode_channels']
        self.electrode_areas = pd.concat([chip_info_['electrode_shapes']['area'], chip_info_['electrode_channels']],
                                        axis=1).dropna().set_index('channel')

    # Use this to turn on electrodes before plateau detection if I can't get the electrodes to actuate
    def turn_on_selected_electrodes(self, voltage=None, frequency=None, electrode_list=None):
        if electrode_list is None:
            electrode_list = []
        self.dropbot_remote.hv_output_enabled = True
        self.dropbot_remote.hv_output_selected = True

        # Get voltage and frequency from the step, or set yours as well
        if voltage is None:
            self.dropbot_remote.voltage = self.active_step_kwargs[u'microdrop.electrode_controller_plugin'][
                u'Voltage (V)']
        else:
            self.dropbot_remote.voltage = voltage

        if frequency is None:
            self.dropbot_remote.frequency = self.active_step_kwargs[u'microdrop.electrode_controller_plugin'][
                u'Frequency (Hz)']
        else:
            self.dropbot_remote.frequency = frequency

        # Create an array of available channels (for dropbot 3 is 120 channels)
        state = np.zeros(self.dropbot_remote.number_of_channels)

        # Turn desired electrodes on
        state[electrode_list] = 1
        self.dropbot_remote.state_of_channels = state


    def get_activated_paths(self):
        # Get activated electrodes in form of electrode###
        electrode_controller_states = self.active_step_kwargs[u'microdrop.electrode_controller_plugin']

        # electrode_list = electrode_controller_states.get('electrode_states', pd.Series())
        path_list = electrode_controller_states['electrode_states'].index
        # Change from electrode list to channel list
        # app = get_app()
        return path_list

    def get_activated_electrodes(self):
        # Get activated electrodes in form of electrode###
        #electrode_controller_states = self.active_step_kwargs[u'microdrop.electrode_controller_plugin']

        #electrode_list = electrode_controller_states.get('electrode_states', pd.Series())

        states = self.active_step_kwargs[u'microdrop.electrode_controller_plugin'].get(u'electrode_states', None)
        if states is None:
            return
        # logger.info("-"*72)
        # logger.info("All stuff: {}".format(states))
        # logger.info("-"*72)
        app = get_app()
        channels = app.dmf_device.df_electrode_channels.set_index('electrode_id')['channel'].drop_duplicates()
        channels = channels.loc[states.index].tolist()
        # logger.info("Should be electrodes: {}".format(channels))
        # logger.info("-"*72)
        return channels

        # Change from electrode list to channel list
        # app = get_app()

        return electrode_list

    def get_state(self, electrode_states):
        app = get_app()

        electrode_channels = (app.dmf_device
                              .actuated_channels(electrode_states.index)
                              .dropna().astype(int))

        # Each channel should be represented *at most* once in
        # `channel_states`.
        channel_states = pd.Series(electrode_states
                                   .ix[electrode_channels.index].values,
                                   index=electrode_channels)
        # Duplicate entries may result from multiple electrodes mapped to the
        # same channel or vice versa.
        channel_states = self.drop_duplicates_by_index(channel_states)
        return channel_states

    def drop_duplicates_by_index(self, series):
        '''
        Drop all but first entry for each set of entries with the same index value.

        Args:

            series (pandas.Series) : Input series.

        Returns:

            (pandas.Series) : Input series with *first* value in `series` for each
                *distinct* index value (i.e., duplicate entries dropped for same
                index value).
        '''
        return series[~series.index.duplicated()]

    def start_monitor(self):
        '''
        .. versionadded:: 2.38

        Start DropBot connection monitor task.
        '''
        if self.monitor_task is not None:
            self.monitor_task.cancel()
            self.control_board = None
            self.dropbot_connected.clear()

        @asyncio.coroutine
        def dropbot_monitor(*args):
            try:
                yield asyncio.From(db.monitor.monitor(*args))
            except asyncio.CancelledError:
                _L().info('Stopped DropBot monitor.')

        self.monitor_task = cancellable(dropbot_monitor)
        thread = threading.Thread(target=self.monitor_task,
                                  args=(self.dropbot_signals, ))
        thread.daemon = True
        thread.start()

    def stop_monitor(self):
        '''
        .. versionadded:: 2.38

        Stop DropBot connection monitor task.
        '''
        if self.dropbot_connected.is_set():
            self.dropbot_connected.clear()
            if self.control_board is not None:
                self.control_board.hv_output_enabled = False
            self.control_board = None
            self.monitor_task.cancel()
            self.monitor_task = None
            self.dropbot_status.on_disconnected()


PluginGlobals.pop_env()
