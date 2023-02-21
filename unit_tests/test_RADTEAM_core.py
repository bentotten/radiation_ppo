import inc_dec    # The code to test
import pytest

import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADTEAM_core

   
class TestIntensityEstimator:    
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
        estimator.update(key=RADTEAM_core.Point((1, 2)), value=300)        
        estimator.reset()
        assert isinstance(locals()['estimator'], RADTEAM_core.IntensityEstimator)
        assert estimator.check_key(RADTEAM_core.Point((1, 2))) == False
        with pytest.raises(ValueError):
            estimator.get_estimate(key=RADTEAM_core.Point((1, 2)))
        assert estimator.get_max() == 0.0
        assert estimator.get_min() == 0.0
