import pytest

import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADTEAM_core
import numpy as np
   
class Test_IntensityEstimator:    
    def test_Update(self)-> None:
        ''' 
            Test update function.
            Should add values to buffer according to the coordinate key
        '''        
        estimator = RADTEAM_core.IntensityEstimator()
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=1000)  
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
            estimator.get_buffer(key=RADTEAM_core.Point((1, 2)))

        # Get buffer
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=1000)
        test_buffer: list = estimator.get_buffer(key=RADTEAM_core.Point((1, 2)))
        assert len(test_buffer) == 1        
        assert test_buffer[0] == 1000
        
        # Add another
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=2000)
        test_buffer2: list = estimator.get_buffer(key=RADTEAM_core.Point((1, 2)))
        assert len(test_buffer2) == 2
        assert test_buffer2[0] == 1000
        assert test_buffer2[1] == 2000
        
        # Add different coordinate
        estimator.update(key=RADTEAM_core.Point((3, 3)), value=350)
        test_buffer2_2: list = estimator.get_buffer(key=RADTEAM_core.Point((1, 2)))
        assert len(test_buffer2_2) == 2
        assert test_buffer2_2[0] == 1000
        assert test_buffer2_2[1] == 2000
        test_buffer3: list = estimator.get_buffer(key=RADTEAM_core.Point((3, 3)))
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
            estimator.get_estimate(key=RADTEAM_core.Point((1, 2)))
        
        # Test median
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=1000)
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=2000)
        median: float = estimator.get_estimate(key=RADTEAM_core.Point((1,2)))
        assert median == 1500
        
        # Add another value
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=500)
        median2: float = estimator.get_estimate(key=RADTEAM_core.Point((1,2)))
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
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=1000)
        assert estimator.get_max() == 1000
        assert estimator.get_min() == 1000
                
        # Test new max update for same location
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=2000)
        assert estimator.get_max() == 1500
        assert estimator.get_min() == 1000
        
        # Test new min update for same location
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=300)
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=300)        
        assert estimator.get_max() == 1500
        assert estimator.get_min() == 650        
        
        # Test min update for new location
        estimator.update(key=RADTEAM_core.Point((3, 3)), value=50)
        assert estimator.get_max() == 1500
        assert estimator.get_min() == 50
        
        # Test max update for new location
        estimator.update(key=RADTEAM_core.Point((4, 4)), value=3000)
        assert estimator.get_max() == 3000
        assert estimator.get_min() == 50        

    def test_CheckKey(self)-> None:
        ''' 
            Test check key function. Should return true if key exists and false if key does not
        '''        
        estimator = RADTEAM_core.IntensityEstimator()   
        assert estimator.check_key(RADTEAM_core.Point((1, 1))) == False
        estimator.update(key=RADTEAM_core.Point((4, 4)), value=3000)
        assert estimator.check_key(RADTEAM_core.Point((1, 1))) == False
        assert estimator.check_key(RADTEAM_core.Point((4, 4))) == True
        
    def test_reset(self)-> None:
        ''' 
            Test reset function. Should reset to a new class object
        '''                
        estimator = RADTEAM_core.IntensityEstimator()
        baseline = RADTEAM_core.IntensityEstimator()
        assert estimator is not baseline
        
        baseline_list = [a for a in dir(baseline) if not a.startswith('__') and not callable(getattr(baseline, a))]

        # Add values        
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=300)        
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
        with pytest.raises(Warning):
            normalizer.normalize_incremental_logscale(current_value=10.0, base=100, increment_value=2)        

        with pytest.raises(Warning):
            normalizer.normalize_incremental_logscale(current_value=10.0, base=10, increment_value=1)        
            
            
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
        tools.readings.update(value=1500, key=RADTEAM_core.Point((1,1)))
        tools.standardizer.update(1500)
        
        tools.reset()

        # Immediate members
        for baseline_att, tools_att in zip(baseline_list, [a for a in dir(tools) if not a.startswith('__') and not callable(getattr(tools, a))]):
            assert getattr(tools, tools_att) == getattr(baseline, baseline_att)
            
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

        test_observation: dict = {0: np.array([1500, 0.5, 0.5, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 1: np.array([1000, 0.6, 0.6, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])}
        for observation in test_observation.values():
            key: RADTEAM_core.Point = RADTEAM_core.Point((observation[1], observation[2]))
            intensity: np.floating = observation[0]
            maps.tools.readings.update(key=key, value=float(intensity))        
        
        _ = maps.observation_to_map(id=0, observation=test_observation)
        
        assert maps.tools.reset_flag == 1
        
        maps.reset()
        
        # Immediate members
        for baseline_att, map_att in zip(baseline_list, [a for a in dir(maps) if not a.startswith('__') and not callable(getattr(maps, a))]):
            test = type(getattr(maps, map_att))
            if test is not RADTEAM_core.ConversionTools:
                if test == np.ndarray:
                    assert getattr(maps, map_att).max() == getattr(baseline, baseline_att).max()
                    assert getattr(maps, map_att).min() == getattr(baseline, baseline_att).min()                
                else:
                    assert getattr(maps, map_att) == getattr(baseline, baseline_att)
            
        # Stored class objects
        assert maps.tools.reset_flag == 2
                
    def test_clear_maps(self, init_parameters)-> None:
        ''' Reset and clear all maps without clearing the observation buffer '''
        maps = RADTEAM_core.MapsBuffer(**init_parameters)
        baseline = RADTEAM_core.MapsBuffer(**init_parameters)
        baseline_list = [a for a in dir(baseline) if not a.startswith('__') and not callable(getattr(baseline, a))]

        test_observation: dict = {0: np.array([1500, 0.5, 0.5, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]), 1: np.array([1000, 0.6, 0.6, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])}
        for observation in test_observation.values():
            key: RADTEAM_core.Point = RADTEAM_core.Point((observation[1], observation[2]))
            intensity: np.floating = observation[0]
            maps.tools.readings.update(key=key, value=float(intensity))        
        
        mapstack = maps.observation_to_map(id=0, observation=test_observation)
        maps.observation_buffer.append([test_observation[0], mapstack]) # TODO Needs better way of matching observation to map_stack
        
        assert maps.tools.reset_flag == 1
        
        maps.clear_maps()
        
        # Immediate members
        for baseline_att, map_att in zip(baseline_list, [a for a in dir(maps) if not a.startswith('__') and not callable(getattr(maps, a))]):
            test = type(getattr(maps, map_att))
            if test is not RADTEAM_core.ConversionTools:
                if map_att == 'observation_buffer':
                    obs_buffer = getattr(maps, map_att)
                    assert np.array_equal(obs_buffer[0][0], test_observation[0])
                    assert len(obs_buffer[0][1]) == 5                   
                    for i, map in enumerate(obs_buffer[0][1]):
                        assert np.array_equal(map, mapstack[i])

                elif test == np.ndarray:
                    assert getattr(maps, map_att).max() == getattr(baseline, baseline_att).max()
                    assert getattr(maps, map_att).min() == getattr(baseline, baseline_att).min()                
                else:
                    assert getattr(maps, map_att) == getattr(baseline, baseline_att) 
                    
        # Stored class objects
        assert maps.tools.reset_flag == 2                    
                 
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
        single_observation: np.ndarray = np.array([1500,  18, 10, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        single_point =(single_observation[1], single_observation[2])
        maps = RADTEAM_core.MapsBuffer(**init_parameters)

        test = maps._deflate_coordinates(single_observation)
        test2 = maps._deflate_coordinates(single_point)
        
        assert test[0] == 0.86 and test[1] ==  0.45636363636363636
        assert test2[0] == 0.86 and test2[1] ==  0.45636363636363636
        
        with pytest.raises(ValueError):
            maps._inflate_coordinates(single_observation[1])

    def test_update_current_agent_location_map(self):
        #, current_coordinates: Tuple[int, int], last_coordinates: Union[Tuple[int, int], None])-> None:
        pass
    
    def test_update_other_agent_locations_map(self):
        #, id: int, current_coordinates: Tuple[int, int], last_coordinates: Union[Tuple[int, int], None])-> None:
        pass

    def test_update_readings_map(self):
        #, coordinates: Tuple[int, int], key: Point)-> None:
        pass

    def _update_visits_count_map(self):
        #, coordinates: Tuple[int, int])-> None:
        pass

    def test_update_obstacle_map(self):
        #, coordinates: Tuple[int, int], single_observation: np.ndarray)-> None:
        pass