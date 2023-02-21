import pytest

import algos.multiagent.NeuralNetworkCores.RADTEAM_core as RADTEAM_core

   
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

        # Min greater than current 
        with pytest.raises(AssertionError):
            normalizer.normalize(current_value=10, max=100, min=11)
                   
        # Processing without min, regular, zero, and negative values
        assert normalizer.normalize(current_value=50, max=100) == 0.5
        assert normalizer.normalize(current_value=-501.0, max=100) == 0.0
        assert normalizer.normalize(current_value=0, max=100) == 0.0        
        
        # Process with min, regular, zero, and negative values
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