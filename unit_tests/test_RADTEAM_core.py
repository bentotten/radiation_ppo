import pytest

import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADTEAM_core

import numpy as np
import torch
import warnings
import time
   
class Test_IntensityEstimator:    
    def test_Update(self)-> None:
        ''' 
            Test update function.
            Should add values to buffer according to the coordinate key
        '''        
        estimator = RADTEAM_core.IntensityEstimator()
        estimator.update(key=(1, 2), value=1000)  
        assert (1, 2) in estimator.readings.keys()
        assert [1000] in estimator.readings.values()
    
    def test_GetBuffer(self)-> None:
        ''' 
            Test get buffer function
            Should pull values into a list from a buffer
        '''        
        estimator = RADTEAM_core.IntensityEstimator()
        
        # Non-existant key
        with pytest.raises(ValueError):
            estimator.get_buffer(key=(1, 2))

        # Get buffer
        estimator.update(key=(1, 2), value=1000)
        test_buffer: list = estimator.get_buffer(key=(1, 2))
        assert len(test_buffer) == 1        
        assert test_buffer[0] == 1000
        
        # Add another
        estimator.update(key=(1, 2), value=2000)
        test_buffer2: list = estimator.get_buffer(key=(1, 2))
        assert len(test_buffer2) == 2
        assert test_buffer2[0] == 1000
        assert test_buffer2[1] == 2000
        
        # Add different coordinate
        estimator.update(key=(3, 3), value=350)
        test_buffer2_2: list = estimator.get_buffer(key=(1, 2))
        assert len(test_buffer2_2) == 2
        assert test_buffer2_2[0] == 1000
        assert test_buffer2_2[1] == 2000
        test_buffer3: list = estimator.get_buffer(key=(3, 3))
        assert len(test_buffer3) == 1        
        assert test_buffer3[0] == 350
        
    def test_GetEstimate(self)-> None:
        ''' 
            Test get median function.
            Should take the median of the existing values stored in a single buffers location
        '''
        estimator = RADTEAM_core.IntensityEstimator()

        # Non-existant key
        with pytest.raises(ValueError):
            estimator.get_estimate(key=(1, 2))
        
        # Test median
        estimator.update(key=(1, 2), value=1000)
        estimator.update(key=(1, 2), value=2000)
        median: float = estimator.get_estimate(key=(1,2))
        assert median == 1500
        
        # Add another value
        estimator.update(key=(1, 2), value=500)
        median2: float = estimator.get_estimate(key=(1,2))
        assert median2 == 1000

    def test_GetMinMax(self)-> None:
        ''' 
            Test get max and get min functions. Should update with latest estimate of true radiation value at that location
            NOTE: the max/min is the ESTIMATE of the true value, not the observed value.
            Should properly update values as more observations are added to the buffers
        '''        
        estimator = RADTEAM_core.IntensityEstimator()        
        
        # Test initial values
        assert estimator.get_max() == 0.0
        assert estimator.get_min() == 0.0
        
        # Test first update
        estimator.update(key=(1, 2), value=1000)
        assert estimator.get_max() == 1000
        assert estimator.get_min() == 1000
                
        # Test new max update for same location
        estimator.update(key=(1, 2), value=2000)
        assert estimator.get_max() == 1500
        assert estimator.get_min() == 1000
        
        # Test new min update for same location
        estimator.update(key=(1, 2), value=300)
        estimator.update(key=(1, 2), value=300)        
        assert estimator.get_max() == 1500
        assert estimator.get_min() == 650        
        
        # Test min update for new location
        estimator.update(key=(3, 3), value=50)
        assert estimator.get_max() == 1500
        assert estimator.get_min() == 50
        
        # Test max update for new location
        estimator.update(key=(4, 4), value=3000)
        assert estimator.get_max() == 3000
        assert estimator.get_min() == 50        

    def test_CheckKey(self)-> None:
        ''' 
            Test check key function. Should return true if key exists and false if key does not
        '''        
        estimator = RADTEAM_core.IntensityEstimator()   
        assert estimator.check_key((1, 1)) == False
        estimator.update(key=(4, 4), value=3000)
        assert estimator.check_key((1, 1)) == False
        assert estimator.check_key((4, 4)) == True
        
    def test_reset(self)-> None:
        ''' 
            Test reset function. Should reset to a new class object
        '''                
        estimator = RADTEAM_core.IntensityEstimator()
        baseline = RADTEAM_core.IntensityEstimator()
        assert estimator is not baseline
        
        baseline_list = [a for a in dir(baseline) if not a.startswith('__') and not callable(getattr(baseline, a))]

        # Add values        
        estimator.update(key=(1, 2), value=300)        
        estimator.reset()
        
        for baseline_att, estimator_att in zip(baseline_list, [a for a in dir(estimator) if not a.startswith('__') and not callable(getattr(estimator, a))]):
            assert getattr(estimator, estimator_att) == getattr(baseline, baseline_att)


class Test_StatisticStandardization:
    def test_Update(self)-> None:
        ''' Test the update function. Should update the running statistics correctly '''
        stats = RADTEAM_core.StatisticStandardization()
        
        # Invalid reading
        with pytest.raises(AssertionError):
            stats.update(reading=-1.0)                
        
        # Set initial mean
        stats.update(reading=1000.0)
        assert stats.mean == 1000.0
        assert stats.count == 1
        assert stats._max == 0
        assert stats._min == 0
        
        # Set next parameter that sets new max
        stats.update(reading=2000.0)
        assert stats.count == 2
        assert stats.mean == 1500.0
        assert stats.square_dist_mean == 500000
        assert stats.sample_variance == 500000
        assert stats.std == pytest.approx(707.10678)        
        assert stats._max == pytest.approx(0.70710678)
        assert stats._min == 0        
   
        # Set next parameter that sets new min
        stats.update(reading=100.0)
        assert stats.count == 3
        assert stats.mean == pytest.approx(1033.33333)
        assert stats.square_dist_mean == pytest.approx(1806666.66666)
        assert stats.sample_variance == pytest.approx(903333.33333)
        assert stats.std == pytest.approx(950.43849)        
        assert stats._max == pytest.approx(0.70710678)
        assert stats._min == pytest.approx(-0.9820028733646521)
        
    def test_Standardize(self):
        ''' Test the standardize function. Should standardize with running statistics correctly '''
        stats = RADTEAM_core.StatisticStandardization()
        
        # Invalid reading
        with pytest.raises(AssertionError):
            stats.standardize(reading=-1.0)                
        
        # Set initial mean
        stats.update(reading=1000.0)
        assert stats.standardize(1) == -999
        assert stats.standardize(1000) == 0
        assert stats.standardize(10000) == 9000
        
        # Set next parameter that sets new max
        stats.update(reading=2000.0)
        assert stats.standardize(1) == pytest.approx(-2.11990612999)
        assert stats.standardize(1000) == pytest.approx(-0.7071067811865475)                   
        assert stats.standardize(10000) == pytest.approx(12.020815280171307) 
        
        # Make sure min and max are not updated during standardize function
        assert stats._max == pytest.approx(0.70710678)
        assert stats._min == 0                 

    def test_GetMaxMin(self):
        ''' Test the get max and min functions. Should get the correct max/min '''
        stats = RADTEAM_core.StatisticStandardization()        
        # Set initial mean
        stats.update(reading=1000.0)
        assert stats.get_max() == 0
        assert stats.get_min() == 0
        
        # Set next parameter that sets new max
        stats.update(reading=2000.0)
        assert stats.get_max() == pytest.approx(0.70710678)
        assert stats.get_min() == 0        
   
        # Set next parameter that sets new min
        stats.update(reading=100.0)
        assert stats.get_max() == pytest.approx(0.70710678)
        assert stats.get_min() == pytest.approx(-0.9820028733646521)        
    
    def test_Reset(self):
        ''' Test the reset function. Should reset correctly to default'''    
        stats = RADTEAM_core.StatisticStandardization()      
        baseline = RADTEAM_core.StatisticStandardization()      
        baseline_list = [a for a in dir(baseline) if not a.startswith('__') and not callable(getattr(baseline, a))]
          
        # Add values        
        stats.update(reading=1000.0)
        stats.update(reading=2000.0)
        stats.update(reading=100)
   
        # Set next parameter that sets new min
        stats.reset()
        
        for baseline_att, stats_att in zip(baseline_list, [a for a in dir(stats) if not a.startswith('__') and not callable(getattr(stats, a))]):
            assert getattr(stats, stats_att) == getattr(baseline, baseline_att)
            
            
class Test_Normalizer:
    def test_Normalize(self):
        ''' Test the normalization function. Should put between range of [0,1]'''
        normalizer = RADTEAM_core.Normalizer()
        
        # Negative max, 0 max, or max that is smaller than current
        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=1.0, max=0.0)

        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=1.0, max=-1.0)
            
        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=100, max=1.0)
            
        # Test negative current with zero max
        assert normalizer.normalize(current_value=-50.0, max=0.0) == 0
        
        # Min greater than current 
        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=10, max=100, min=11)
                   
        # Processing without min, regular, zero, and negative values for current
        assert normalizer.normalize(current_value=50, max=100) == 0.5
        assert normalizer.normalize(current_value=-501.0, max=100) == 0.0
        assert normalizer.normalize(current_value=0, max=100) == 0.0        
        
        # Process with min, regular, zero, and negative values for min
        assert normalizer.normalize(current_value=50, max=100, min=10) == pytest.approx(0.4444444444444444)
        assert normalizer.normalize(current_value=50, max=100, min=-10) == pytest.approx(0.5454545454545454)
        assert normalizer.normalize(current_value=0, max=100, min=-10) == 0.0
        assert normalizer.normalize(current_value=50, max=100, min=0) == pytest.approx(0.5)

    def test_LogNormalize(self):
        ''' Test the normalization function. Should put between range of [0,1]'''
        normalizer = RADTEAM_core.Normalizer()
        
        # Test invalid inputs
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=-1.0, base=10)
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=1.0, base=-10)
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=1.0, base=0)            
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=1.0, base=10, increment_value=0)
        with pytest.raises(AssertionError):            
            normalizer.normalize_incremental_logscale(current_value=1.0, base=10, increment_value=-1)         

        # Test normal
        assert normalizer.normalize_incremental_logscale(current_value=4.0, base=10, increment_value=2) == pytest.approx(0.598104004)
        assert normalizer.normalize_incremental_logscale(current_value=4.0, base=10, increment_value=2) == ( 
                                                pytest.approx(normalizer.normalize_incremental_logscale(current_value=4.0, base=10)) )
        # Test Max
        assert normalizer.normalize_incremental_logscale(current_value=18.0, base=10, increment_value=2) == 1

        # Test realistic min
        assert normalizer.normalize_incremental_logscale(current_value=1.0, base=10, increment_value=2) == pytest.approx(0.366725791)
        
        # Test assert fail for out of boundaries
        with pytest.raises(AssertionError):
            normalizer.normalize_incremental_logscale(current_value=30.0, base=10, increment_value=2)
            
        # Test warning for change of base or increment value
        #with pytest.raises(Warning):
        with warnings.catch_warnings():        
            normalizer.normalize_incremental_logscale(current_value=10.0, base=100, increment_value=2)        

        with warnings.catch_warnings():        
            normalizer.normalize_incremental_logscale(current_value=10.0, base=10, increment_value=10)        
            
            
class Test_ConversionTools:
    def test_Init(self):
        ''' Test the conversion tool initialization. Should initialize all desired objects'''

        tools = RADTEAM_core.ConversionTools()
        
        assert isinstance(tools.last_coords, dict)
        assert isinstance(tools.readings, RADTEAM_core.IntensityEstimator)
        assert isinstance(tools.standardizer, RADTEAM_core.StatisticStandardization)

    def test_Reset(self)-> None:
        ''' Reset and clear all members '''
        tools = RADTEAM_core.ConversionTools()
        baseline = RADTEAM_core.ConversionTools()
        baseline_list = [a for a in dir(baseline) if not a.startswith('__') and not callable(getattr(baseline, a))]
        
        baseline_readings = [a for a in dir(baseline.readings) if not a.startswith('__') and not callable(getattr(baseline.readings, a))]
        baseline_standardizer = [a for a in dir(baseline.standardizer) if not a.startswith('__') and not callable(getattr(baseline.standardizer, a))]

        tools.last_coords[(1, 1)] = [30, 20, 10] # type: ignore
        tools.readings.update(value=1500, key=(1,1))
        tools.standardizer.update(1500)
        
        tools.reset()

        # Immediate members
        for baseline_att, tools_att in zip(baseline_list, [a for a in dir(tools) if not a.startswith('__') and not callable(getattr(tools, a))]):
            if tools_att != 'reset_flag':
                assert getattr(tools, tools_att) == getattr(baseline, baseline_att)
            else:
                assert getattr(tools, tools_att) == 1
            
        # Stored class objects
        for baseline_att, tools_att in zip(baseline_readings, [a for a in dir(tools.readings) if not a.startswith('__') and not callable(getattr(tools.readings, a))]):
            assert getattr(tools.readings, tools_att) == getattr(baseline.readings, baseline_att)
            
        for baseline_att, tools_att in zip(baseline_standardizer, [a for a in dir(tools.standardizer) if not a.startswith('__') and not callable(getattr(tools.standardizer, a))]):
            assert getattr(tools.standardizer, tools_att) == getattr(baseline.standardizer, baseline_att)
            

class Test_MapBuffer:
    @pytest.fixture
    def init_parameters(self)-> dict:
        ''' Set up initialization parameters for mapbuffer '''
        return dict(
            observation_dimension=11,
            steps_per_episode=120,
            number_of_agents=2
        )
    
    def test_Init(self, init_parameters):
        ''' Test the Map Buffer initialization. Should initialize all desired objects'''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        assert isinstance(maps.tools, RADTEAM_core.ConversionTools)

    def test_Reset(self, init_parameters)-> None:
        ''' Reset and clear all members '''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        baseline = RADTEAM_core.MapsBuffer(**init_parameters)
        baseline_list = [a for a in dir(baseline) if not a.startswith('__') and not callable(getattr(baseline, a))]

        # Setup
        test_observation: dict = {0: np.array([1500, 0.5, 0.5, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 1: np.array([1000, 0.6, 0.6, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])}
        for observation in test_observation.values():
            key =(observation[1], observation[2])
            intensity: np.floating = observation[0]
            maps.tools.readings.update(key=key, value=float(intensity))        
        
        _ = maps.observation_to_map(id=0, observation=test_observation)
        
        assert maps.tools.reset_flag == 1
        
        # Test full reset
        start_time = time.time()           
        maps.full_reset()
        full_reset_time = time.time() - start_time
        
        # Immediate members
        for baseline_att, map_att in zip(baseline_list, [a for a in dir(maps) if not a.startswith('__') and not callable(getattr(maps, a))]):
            test = type(getattr(maps, map_att))
            if test is not RADTEAM_core.ConversionTools:
                if test == np.ndarray:
                    assert getattr(maps, map_att).max() == getattr(baseline, baseline_att).max()
                    assert getattr(maps, map_att).min() == getattr(baseline, baseline_att).min()                
                else:
                    assert getattr(maps, map_att) == getattr(baseline, baseline_att)
            
        assert maps.tools.reset_flag == 2
        
        # Test end-of-epoch reset, where matrices are cleared but not reinstatiated
        _ = maps.observation_to_map(id=0, observation=test_observation)
        
        start_time = time.time()           
        maps.reset()
        reset_time = time.time() - start_time        
        
        assert (full_reset_time + 0.01) > reset_time
        
        # Immediate members
        for baseline_att, map_att in zip(baseline_list, [a for a in dir(maps) if not a.startswith('__') and not callable(getattr(maps, a))]):
            test = type(getattr(maps, map_att))
            if test is not RADTEAM_core.ConversionTools:
                if test == np.ndarray:
                    assert getattr(maps, map_att).max() == getattr(baseline, baseline_att).max()
                    assert getattr(maps, map_att).min() == getattr(baseline, baseline_att).min()                
                else:
                    assert getattr(maps, map_att) == getattr(baseline, baseline_att)
            
        assert maps.tools.reset_flag == 3        
                 
    def test_inflate_coordinates(self, init_parameters)-> None:
        ''' Test coordinate inflation for both observation and point '''
        single_observation: np.ndarray = np.array([1500,  0.86, 0.45636363636363636, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        single_point = RADTEAM_core.Point((single_observation[1], single_observation[2]))
        maps = RADTEAM_core.MapsBuffer(**init_parameters)

        test = maps._inflate_coordinates(single_observation)
        test2 = maps._inflate_coordinates(single_point)
        
        assert test[0] == 18 and test[1] == 10
        assert test2[0] == 18 and test2[1] == 10
        
        with pytest.raises(ValueError):
            maps._inflate_coordinates(single_observation[1])
        
    def test_deflate_coordinates(self, init_parameters)-> None:
        ''' Test coordinate deflation for both observation and point '''        
        single_observation: np.ndarray = np.array([1500, 18.92, 10.04, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        single_point =(single_observation[1], single_observation[2])
        maps = RADTEAM_core.MapsBuffer(**init_parameters)

        test = maps._deflate_coordinates(single_observation)
        test2 = maps._deflate_coordinates(single_point)
        
        assert test[0] == pytest.approx(0.86) and test[1] == pytest.approx(0.45636363636363636)
        assert test2[0] == pytest.approx(0.86) and test2[1] == pytest.approx(0.45636363636363636)
        
        with pytest.raises(ValueError):
            maps._inflate_coordinates(single_observation[1])

    def test_update_current_agent_location_map(self, init_parameters):
        ''' Test current agent location map update '''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        
        # Test normal first update
        maps._update_current_agent_location_map(current_coordinates=(0, 1), last_coordinates=None)
        assert maps.location_map[0][1] == 1.0
        flat = np.delete(maps.location_map.ravel(), 1)
        assert flat.max() == 0.0 
            
        with pytest.raises(AssertionError):
            maps.location_map[2][2] = 2
            maps._update_current_agent_location_map(current_coordinates=(0, 1), last_coordinates=(0, 1))  # Maximum value exceeded      
    
    def test_update_other_agent_locations_map(self, init_parameters):
        ''' Test other agent locations map update '''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        
        # Test normal first update
        maps._update_other_agent_locations_map(current_coordinates=(0, 1), last_coordinates=None)
        assert maps.others_locations_map[0][1] == 1.0
        flat = np.delete(maps.others_locations_map.ravel(), 1)
        assert flat.max() == 0.0
        
        maps._update_other_agent_locations_map(current_coordinates=(0, 2), last_coordinates=(0, 1))
        assert maps.others_locations_map[0][2] == 1.0
        flat = maps.others_locations_map.ravel()
        flat_t1 = np.delete(flat, 2)
        flat_t2 = np.delete(flat_t1, 1)
        assert flat_t2.max() == 0.0
        
    def test_update_combined_agent_locations_map(self, init_parameters):
        ''' Test other agent locations map update '''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        
        # Test normal first update
        maps._update_combined_agent_locations_map(current_coordinates=(0, 1), last_coordinates=None)
        assert maps.combined_location_map[0][1] == 1.0
        flat = np.delete(maps.combined_location_map.ravel(), 1)
        assert flat.max() == 0.0
        
        maps._update_combined_agent_locations_map(current_coordinates=(0, 2), last_coordinates=(0, 1))
        assert maps.combined_location_map[0][2] == 1.0
        flat = maps.combined_location_map.ravel()
        flat_t1 = np.delete(flat, 2)
        flat_t2 = np.delete(flat_t1, 1)
        assert flat_t2.max() == 0.0        

    def test_update_readings_map(self, init_parameters)-> None:
        ''' test method to update the radiation intensity observation map. If prior location exists, this is overwritten with the latest estimation. '''

        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        coords_1 = (0, 1)
        coords_2 = (0, 2)
        
        observation =  {0: [1000.0, coords_1[0], coords_1[1]], 1: [200, coords_2[0], coords_2[1]], 2: [1200.0, coords_1[0], coords_1[1]]}
        
        for obs in observation.values():
            key = (obs[1], obs[2]) # type: ignore
            intensity: np.floating = obs[0] # type: ignore
            maps.tools.readings.update(key=key, value=float(intensity))              
        
        # Test initial updates
        maps._update_readings_map(coordinates=coords_1)
        assert maps.readings_map[coords_1[0]][coords_1[1]] == 0.0 # First reading is always 0
        maps._update_readings_map(coordinates=coords_2)
        assert maps.readings_map[coords_2[0]][coords_2[1]] == 0.0 # New minimum is always 0 after offset       
        maps._update_readings_map(coordinates=coords_2)
        assert maps.readings_map[coords_2[0]][coords_2[1]] == pytest.approx(0.18350341907227394)
        maps._update_readings_map(coordinates=coords_1)
        assert maps.readings_map[coords_1[0]][coords_1[1]] == 1.0          
        
        # Test with valid key 
        maps._update_readings_map(coordinates=coords_2, key=coords_2)
        assert maps.readings_map[coords_2[0]][coords_2[1]] == 0.0 # New minimum is always 0 after offset       
        
        # Test invalid key
        with pytest.raises(ValueError):
            maps._update_readings_map(coordinates=(0, 1), key=(0, 0))

    def test_update_visits_count_map(self, init_parameters):
        ''' Test method to update the visits count observation map. Increments in a logarithmic fashion. '''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        
        # Test normal update
        maps._update_visits_count_map(coordinates=(0, 1))
        assert maps.visit_counts_map[0][1] == pytest.approx(0.11212191)
        assert maps.visit_counts_shadow[(0,1)] == 2
        
        maps._update_visits_count_map(coordinates=(0, 2))
        assert maps.visit_counts_map[0][1] == pytest.approx(0.11212191)
        assert maps.visit_counts_shadow[(0,1)] == 2        
        assert maps.visit_counts_map[0][2] == pytest.approx(0.11212191)
        assert maps.visit_counts_shadow[(0,2)] == 2
        
        maps._update_visits_count_map(coordinates=(0, 2))
        assert maps.visit_counts_map[0][1] == pytest.approx(0.11212191)
        assert maps.visit_counts_shadow[(0,1)] == 2        
        assert maps.visit_counts_map[0][2] == pytest.approx(0.22424382)
        assert maps.visit_counts_shadow[(0,2)] == 4
                
        # Test will never go above one for max steps in same location
        for _ in range(init_parameters['number_of_agents'] * init_parameters['steps_per_episode']-1):
            maps._update_visits_count_map(coordinates=(0, 3))
            
        with warnings.catch_warnings():        
            maps._update_visits_count_map(coordinates=(0, 3))
        assert maps.visit_counts_map[0][3] == pytest.approx(0.99865758) # NOTE: Will not be exactly one, as entire base needs to be offset by 1 step for reward bootstrapping
        assert maps.visit_counts_shadow[(0,3)] == (2 * init_parameters['number_of_agents'] * init_parameters['steps_per_episode'])
        
    def test_update_obstacle_map(self, init_parameters)-> None:
        ''' Test method to update the obstacle detection observation map. Renders headmap of Agent distance from obstacle '''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        
        single_observation: np.ndarray = np.array([1500,  0.86, 0.45636363636363636, 0.1, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1])        
        coordinates = (0,1)
        coordinates_2 = (0, 0)
        
        maps._update_obstacle_map(single_observation=single_observation, coordinates=coordinates)
        assert maps.obstacles_map[0][1] == pytest.approx(0.1)
        
        maps._update_obstacle_map(single_observation=single_observation, coordinates=coordinates_2)
        assert maps.obstacles_map[0][1] == pytest.approx(0.1)
        assert maps.obstacles_map[0][0] == pytest.approx(0.1)
        
    def test_observation_to_map(self, init_parameters) -> None:
        ''' Note: Actual math is tested in individual map tests. This checks that proper coordinate was updated '''
        
        init_parameters['number_of_agents'] = 3
        
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        
        ## Test first update   
        step1 = maps._deflate_coordinates((0, 1))
        step2 = maps._deflate_coordinates((0, 2))
        
        observations = {
            0: np.array([1000.0, step1[0], step1[1], 0., 0., 0., 0.1, 0., 0., 0., 0.], dtype=np.float32), 
            1: np.array([1000.0, step1[0], step1[1], 0., 0., 0., 0.1, 0., 0., 0., 0.], dtype=np.float32),
            2: np.array([1000.0, step2[0], step2[1], 0., 0., 0., 0.1, 0., 0., 0., 0.], dtype=np.float32)            
            }     
            
        mapstack = maps.observation_to_map(observation=observations, id=0)      
        
        # Test Locations map
        assert maps.location_map[0][1] == 1.0
        assert mapstack[0][0][1] == 1.0        
        flat = np.delete(maps.location_map.ravel(), 1)
        assert flat.max() == 0.0
        flat2 = np.delete(mapstack[0].ravel(), 1)
        assert flat2.max() == 0.0     
        
        # Test Other locations map
        assert maps.others_locations_map[0][1] == 1.0
        assert mapstack[1][0][1] == 1.0        
        assert maps.others_locations_map[0][2] == 1.0
        assert mapstack[1][0][2] == 1.0
        flat = maps.others_locations_map.ravel()
        flat_t1 = np.delete(flat, 1)
        flat_t2 = np.delete(flat_t1, 1)
        assert flat_t2.max() == 0.0                 

        # Test Readings map registered - First reading is always 0
        assert maps.readings_map.max() == 0.0 
        assert mapstack[2].max() == 0.0
        
        # Test Visits count
        assert maps.visit_counts_map[0][1] > 0.0
        assert mapstack[3][0][1] > 0.0
        assert mapstack[3][0][1] == maps.visit_counts_map[0][1]     
        assert maps.visit_counts_map[0][2] > 0.0
        assert mapstack[3][0][2] > 0.0
        assert mapstack[3][0][2] == maps.visit_counts_map[0][2]         
        assert maps.visit_counts_map[0][1] > maps.visit_counts_map[0][2]
        
        # Test Obstacle Map
        assert maps.obstacles_map[0][1] > 0.0
        assert mapstack[4][0][1] > 0.0
        assert maps.obstacles_map[0][2] > 0.0
        assert mapstack[4][0][2] > 0.0
        
        # Test combo locations map
        assert maps.combined_location_map[0][1] == 2.0
        assert mapstack[5][0][1] == 2.0        
        assert maps.combined_location_map[0][2] == 1.0
        assert mapstack[5][0][2] == 1.0
        flat = maps.combined_location_map.ravel()
        flat_t1 = np.delete(flat, 1)
        flat_t2 = np.delete(flat_t1, 1)
        assert flat_t2.max() == 0.0               
                
        ## Test second update
        step1 = maps._deflate_coordinates((0, 3))
        step2 = maps._deflate_coordinates((0, 4))
        
        observations = {
            0: np.array([5000.0, step1[0], step1[1], 0., 0., 0., 0.1, 0., 0., 0., 0.], dtype=np.float32), 
            1: np.array([5000.0, step1[0], step1[1], 0., 0., 0., 0.1, 0., 0., 0., 0.], dtype=np.float32),
            2: np.array([5000.0, step2[0], step2[1], 0., 0., 0., 0.1, 0., 0., 0., 0.], dtype=np.float32)            
            }  

        mapstack = maps.observation_to_map(observation=observations, id=0)

        # Test Locations map
        assert maps.location_map[0][1] == 0.0
        assert mapstack[0][0][1] == 0.0        
        assert maps.location_map[0][3] == 1.0
        assert mapstack[0][0][3] == 1.0    
        
        # Test Other locations map
        assert maps.others_locations_map[0][1] == 0.0
        assert mapstack[1][0][1] == 0.0        
        assert maps.others_locations_map[0][2] == 0.0
        assert mapstack[1][0][2] == 0.0     
        assert maps.others_locations_map[0][3] == 1.0
        assert mapstack[1][0][3] == 1.0        
        assert maps.others_locations_map[0][4] == 1.0
        assert mapstack[1][0][4] == 1.0                     

        # Test Readings map registered - Actual math tested in individual unit test   
        assert maps.readings_map[0][3] > 0.0
        assert mapstack[2][0][3] > 0.0     
        assert maps.readings_map[0][4] > 0.0
        assert mapstack[2][0][4] > 0.0     
        
        # Test Visits count
        assert maps.visit_counts_map[0][1] > 0.0
        assert mapstack[3][0][1] > 0.0       
        assert maps.visit_counts_map[0][2] > 0.0
        assert mapstack[3][0][2] > 0.0     
        assert maps.visit_counts_map[0][3] > 0.0
        assert mapstack[3][0][3] > 0.0     
        assert maps.visit_counts_map[0][4] > 0.0
        assert mapstack[3][0][4] > 0.0     
        
        # Test Obstacle Map
        assert maps.obstacles_map[0][1] > 0.0
        assert mapstack[4][0][1] > 0.0       
        assert maps.obstacles_map[0][2] > 0.0
        assert mapstack[4][0][2] > 0.0     
        assert maps.obstacles_map[0][3] > 0.0
        assert mapstack[4][0][3] > 0.0     
        assert maps.obstacles_map[0][4] > 0.0
        assert mapstack[4][0][4] > 0.0 
        
        # Test combo locations map
        assert maps.combined_location_map[0][1] == 0.0
        assert mapstack[5][0][1] == 0.0           
        assert maps.combined_location_map[0][2] == 0.0
        assert mapstack[5][0][2] == 0.0     
        assert maps.combined_location_map[0][3] == 2.0
        assert mapstack[5][0][3] == 2.0                
        assert maps.combined_location_map[0][4] == 1.0
        assert mapstack[5][0][4] == 1.0              
           

class Test_Actor:
    @pytest.fixture
    def init_parameters(self)-> dict:
        ''' Set up initialization parameters for Actor '''
        torch.manual_seed(0)
        np.random.seed(0)
        
        return dict(
            map_dim=(2,2),
            batches=1,
            map_count=5,
            action_dim=8,
        )
        
    @pytest.fixture
    def create_mapstack(self)-> torch.Tensor:
        ''' Set up a mock mapstack '''
        agent_loc = np.zeros((2, 2), dtype=np.float32)
        agent_loc[0][0] = 1.0
        other_loc = np.zeros((2, 2), dtype=np.float32)
        other_loc[0][1] = 1.0
        radiation = np.zeros((2, 2), dtype=np.float32)
        radiation[1][1] = 0.5
        radiation[0][1] = 0.7
        visits = np.zeros((2, 2), dtype=np.float32)
        visits[0][0] = 0.23
        visits[0][1] = 0.23
        obstacles = np.zeros((2, 2), dtype=np.float32)
        obstacles[1][0] = 0.9
        
        map_stack: torch.Tensor = torch.stack(
            [torch.tensor(agent_loc), torch.tensor(other_loc), torch.tensor(radiation), torch.tensor(visits),  torch.tensor(obstacles)]
            )
        
        batched_map_stack: torch.Tensor = torch.unsqueeze(map_stack, dim=0) 
        return batched_map_stack
        
    def test_Init(self, init_parameters, create_mapstack):
        _ = RADTEAM_core.Actor(**init_parameters)

    def test_Layers(self, init_parameters, create_mapstack):
        ''' Test layers are the shapes they should be'''
        pi = RADTEAM_core.Actor(**init_parameters)
        pi.eval()
        mapstack = create_mapstack
        
        for i, layer in enumerate(pi.actor.children()):
            if i == 0:
                # First convolution
                output = layer(mapstack)

                assert output.size() == (1, 8, 2, 2)
            elif i == 1:
                # Relu
                output = layer(output)                
                
                assert output.size() == (1, 8, 2, 2)                
            elif i == 2:
                # Maxpool
                output = layer(output)

                assert output.size() == (1, 8, 1, 1)
            elif i == 3:
                # Second Convolution
                output = layer(output)

                assert output.size() == (1, 16, 1, 1)
            elif i == 4:
                # Relu
                output = layer(output)

                assert output.size() == (1, 16, 1, 1)
            elif i == 5:
                # Flatten layer
                output = layer(output)

                assert output.numel() == 16
            elif i == 6:
                # First linear
                output = layer(output)

                assert output.numel() == 32
            elif i == 7:
                # relu
                output = layer(output)

                assert output.numel() == 32
            elif i == 8:
                # Second Linear
                output = layer(output)

                assert output.numel() == 16
            elif i == 9:
                # relu
                output = layer(output)

                assert output.numel() == 16
            elif i == 10:
                # output linear
                output = layer(output)

                assert output.numel() == 8
            elif i == 11:
                # softmax
                output = layer(output)

                assert output.numel() == 8
            else:            
                raise Exception("Too many layers seen")

    def test_act(self, init_parameters, create_mapstack):
        ''' Test layers are the shapes they should be'''
        pi = RADTEAM_core.Actor(**init_parameters)
        pi.eval()
        mapstack = create_mapstack
        action, logprob = pi.act(mapstack) #TODO test logprob
        assert action >= 0 and action < 8

    def test_modes(self, init_parameters):
        pi = RADTEAM_core.Actor(**init_parameters)
        pi.put_in_training_mode()
        assert pi.actor.training == True
        pi.put_in_evaluation_mode()
        assert pi.actor.training == False


class Test_Critic:
    @pytest.fixture
    def init_parameters(self)-> dict:
        ''' Set up initialization parameters for Critic '''
        torch.manual_seed(0)
        np.random.seed(0)
        
        return dict(
            map_dim=(2,2),
            batches=1,
            map_count=5,
        )
        
    @pytest.fixture
    def create_mapstack(self)-> torch.Tensor:
        ''' Set up a mock mapstack '''
        agent_loc = np.zeros((2, 2), dtype=np.float32)
        agent_loc[0][0] = 1.0
        other_loc = np.zeros((2, 2), dtype=np.float32)
        other_loc[0][1] = 1.0
        radiation = np.zeros((2, 2), dtype=np.float32)
        radiation[1][1] = 0.5
        radiation[0][1] = 0.7
        visits = np.zeros((2, 2), dtype=np.float32)
        visits[0][0] = 0.23
        visits[0][1] = 0.23
        obstacles = np.zeros((2, 2), dtype=np.float32)
        obstacles[1][0] = 0.9
        
        map_stack: torch.Tensor = torch.stack(
            [torch.tensor(agent_loc), torch.tensor(other_loc), torch.tensor(radiation), torch.tensor(visits),  torch.tensor(obstacles)]
            )
        
        batched_map_stack: torch.Tensor = torch.unsqueeze(map_stack, dim=0) 
        return batched_map_stack
        
    def test_Init(self, init_parameters, create_mapstack):
        _ = RADTEAM_core.Critic(**init_parameters)

    def test_Layers(self, init_parameters, create_mapstack):
        ''' Test layers are the shapes they should be'''
        critic = RADTEAM_core.Critic(**init_parameters)
        critic.eval()
        mapstack = create_mapstack
        
        for i, layer in enumerate(critic.critic.children()):
            if i == 0:
                # First convolution
                output = layer(mapstack)
                assert output.size() == (1, 8, 2, 2)
            elif i == 1:
                # Relu
                output = layer(output)                
                assert output.size() == (1, 8, 2, 2)                
            elif i == 2:
                # Maxpool
                output = layer(output)
                assert output.size() == (1, 8, 1, 1)
            elif i == 3:
                # Second Convolution
                output = layer(output)
                assert output.size() == (1, 16, 1, 1)
            elif i == 4:
                # Relu
                output = layer(output)
                assert output.size() == (1, 16, 1, 1)
            elif i == 5:
                # Flatten layer
                output = layer(output)
                assert output.numel() == 16
            elif i == 6:
                # First linear
                output = layer(output)
                assert output.numel() == 32
            elif i == 7:
                # relu
                output = layer(output)
                assert output.numel() == 32
            elif i == 8:
                # Second Linear
                output = layer(output)
                assert output.numel() == 16
            elif i == 9:
                # relu
                output = layer(output)
                assert output.numel() == 16
            elif i == 10:
                # output linear
                output = layer(output)
                assert output.numel() == 1
            else:            
                raise Exception("Too many layers seen")

    def test_forward(self, init_parameters, create_mapstack):
        ''' Test layers are the shapes they should be'''
        critic = RADTEAM_core.Critic(**init_parameters)
        critic.eval()
        mapstack = create_mapstack
        state_value = critic.forward(mapstack)
        test = state_value.numel()
        assert state_value.numel() == 1
        # TODO add better check

    def test_modes(self, init_parameters):
        critic = RADTEAM_core.Critic(**init_parameters)
        critic.put_in_training_mode()
        assert critic.critic.training == True
        critic.put_in_evaluation_mode()
        assert critic.critic.training == False

