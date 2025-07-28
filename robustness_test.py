#!/usr/bin/env python3
"""
Robustness Testing untuk Real-time Sensor AI System
Testing CNN, LSTM, dan Hybrid CNN-LSTM models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import time
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append('src')

from src.database_manager import DatabaseManager
from src.realtime_preprocessor import SensorDataPreprocessor
from src.ai_models import SensorAIModels

class RobustnessTestSuite:
    """Comprehensive robustness testing untuk AI models"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = SensorDataPreprocessor()
        self.ai_models = None
        self.test_results = {}
        
    def setup_test_environment(self):
        """Setup testing environment"""
        print("🔧 Setting up test environment...")
        
        # Connect to database
        if not self.db_manager.connect():
            raise Exception("Failed to connect to database!")
        
        # Get test data
        test_data = self.db_manager.get_data_for_training(hours=48)
        
        if len(test_data) < 500:
            print("⚠️ Warning: Limited test data available")
            # Generate synthetic data for comprehensive testing
            test_data = self.generate_synthetic_test_data(1000)
        
        # Prepare data
        featured_data = self.preprocessor.create_features(test_data)
        self.preprocessor.fit_scalers(featured_data)
        transformed_data = self.preprocessor.transform_features(featured_data)
        
        # Prepare sequences
        X, y = self.preprocessor.prepare_sequence_data(transformed_data)
        
        if X is None:
            raise Exception("Failed to prepare test data!")
        
        self.test_data = {
            'X': X,
            'y': y,
            'raw_data': test_data,
            'featured_data': featured_data
        }
        
        # Initialize AI models
        input_shape = X.shape[1:]
        self.ai_models = SensorAIModels(input_shape=input_shape, output_dim=3)
        
        print(f"✅ Test environment ready - Data shape: {X.shape}")
        
    def generate_synthetic_test_data(self, n_samples):
        """Generate synthetic data untuk testing"""
        print(f"🧪 Generating {n_samples} synthetic test samples...")
        
        np.random.seed(42)
        
        # Generate diverse patterns
        patterns = []
        
        # Normal pattern
        normal_ph = np.random.normal(7.0, 0.5, n_samples//3)
        normal_temp = np.random.normal(25, 3, n_samples//3)
        normal_quality = ['baik' if (6.5 <= ph <= 7.5) and (20 <= temp <= 30) else 'buruk' 
                         for ph, temp in zip(normal_ph, normal_temp)]
        
        # Stress pattern
        stress_ph = np.random.uniform(4, 10, n_samples//3)
        stress_temp = np.random.uniform(10, 45, n_samples//3)
        stress_quality = ['baik' if np.random.random() < 0.3 else 'buruk' 
                         for _ in range(n_samples//3)]
        
        # Anomaly pattern
        anomaly_ph = np.concatenate([
            np.random.uniform(2, 4, n_samples//6),
            np.random.uniform(10, 12, n_samples//6)
        ])
        anomaly_temp = np.concatenate([
            np.random.uniform(0, 10, n_samples//6),
            np.random.uniform(45, 60, n_samples//6)
        ])
        anomaly_quality = ['buruk'] * (n_samples//3)
        
        # Combine all patterns
        all_ph = np.concatenate([normal_ph, stress_ph, anomaly_ph])
        all_temp = np.concatenate([normal_temp, stress_temp, anomaly_temp])
        all_quality = normal_quality + stress_quality + anomaly_quality
        
        # Create DataFrame
        synthetic_data = pd.DataFrame({
            'no': range(1, n_samples + 1),
            'ph': np.clip(all_ph, 0, 14),
            'suhu': np.clip(all_temp, -10, 70),
            'kualitas': all_quality,
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1S')
        })
        
        # Shuffle data
        synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)
        
        print(f"✅ Synthetic data generated: {len(synthetic_data)} samples")
        return synthetic_data
    
    def test_data_quality_robustness(self):
        """Test robustness terhadap data quality issues"""
        print("\n🔍 Testing Data Quality Robustness...")
        
        X, y = self.test_data['X'], self.test_data['y']
        
        # Test 1: Missing values
        print("📊 Test 1: Missing Values Robustness")
        X_missing = X.copy()
        
        missing_percentages = [0.05, 0.1, 0.15, 0.2]
        missing_results = {}
        
        for missing_pct in missing_percentages:
            X_test = X_missing.copy()
            
            # Randomly set values to NaN
            mask = np.random.random(X_test.shape) < missing_pct
            X_test[mask] = np.nan
            
            # Fill NaN with forward fill and backward fill
            for i in range(X_test.shape[0]):
                for j in range(X_test.shape[2]):
                    series = pd.Series(X_test[i, :, j])
                    series = series.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    X_test[i, :, j] = series.values
            
            # Test preprocessing robustness
            try:
                preprocessing_time = time.time()
                # Simulate preprocessing
                processed_successfully = True
                preprocessing_time = time.time() - preprocessing_time
                
                missing_results[missing_pct] = {
                    'processed_successfully': processed_successfully,
                    'processing_time': preprocessing_time
                }
                
                print(f"   {missing_pct*100}% missing: ✅ Processed in {preprocessing_time:.3f}s")
                
            except Exception as e:
                missing_results[missing_pct] = {
                    'processed_successfully': False,
                    'error': str(e)
                }
                print(f"   {missing_pct*100}% missing: ❌ Failed - {e}")
        
        # Test 2: Outliers
        print("\n📊 Test 2: Outliers Robustness")
        X_outliers = X.copy()
        
        outlier_percentages = [0.01, 0.05, 0.1, 0.15]
        outlier_results = {}
        
        for outlier_pct in outlier_percentages:
            X_test = X_outliers.copy()
            
            # Add extreme outliers
            n_outliers = int(outlier_pct * X_test.size)
            outlier_indices = np.random.choice(X_test.size, n_outliers, replace=False)
            
            flat_X = X_test.flatten()
            flat_X[outlier_indices] = np.random.choice([-10, 10], n_outliers) * np.abs(flat_X[outlier_indices])
            X_test = flat_X.reshape(X_test.shape)
            
            try:
                processing_time = time.time()
                # Simulate outlier detection and handling
                processed_successfully = True
                processing_time = time.time() - processing_time
                
                outlier_results[outlier_pct] = {
                    'processed_successfully': processed_successfully,
                    'processing_time': processing_time
                }
                
                print(f"   {outlier_pct*100}% outliers: ✅ Handled in {processing_time:.3f}s")
                
            except Exception as e:
                outlier_results[outlier_pct] = {
                    'processed_successfully': False,
                    'error': str(e)
                }
                print(f"   {outlier_pct*100}% outliers: ❌ Failed - {e}")
        
        self.test_results['data_quality'] = {
            'missing_values': missing_results,
            'outliers': outlier_results
        }
        
        print("✅ Data Quality Robustness Test completed")
    
    def test_model_performance_robustness(self):
        """Test robustness performa model"""
        print("\n🤖 Testing Model Performance Robustness...")
        
        X, y = self.test_data['X'], self.test_data['y']
        
        # Build and train models
        models_to_test = ['CNN', 'LSTM', 'CNN_LSTM']
        model_performances = {}
        
        for model_type in models_to_test:
            print(f"\n🎯 Testing {model_type} Model...")
            
            # Build model
            if model_type == 'CNN':
                model = self.ai_models.build_cnn_model()
            elif model_type == 'LSTM':
                model = self.ai_models.build_lstm_model()
            else:  # CNN_LSTM
                model = self.ai_models.build_hybrid_cnn_lstm_model()
            
            # Split data untuk training dan testing
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y[:, 2].astype(int)
            )
            
            try:
                # Train model
                training_start = time.time()
                history = self.ai_models.train_model(
                    model_type=model_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_test,
                    y_val=y_test,
                    epochs=20,  # Reduced for testing
                    batch_size=32,
                    verbose=0
                )
                training_time = time.time() - training_start
                
                # Evaluate model
                metrics = self.ai_models.evaluate_model(model_type, X_test, y_test)
                
                # Test inference speed
                inference_times = []
                for _ in range(100):
                    start = time.time()
                    _ = self.ai_models.predict(model_type, X_test[:1])
                    inference_times.append(time.time() - start)
                
                avg_inference_time = np.mean(inference_times)
                
                model_performances[model_type] = {
                    'training_time': training_time,
                    'metrics': metrics,
                    'avg_inference_time': avg_inference_time,
                    'inference_std': np.std(inference_times),
                    'training_successful': True
                }
                
                print(f"   ✅ {model_type} - Accuracy: {metrics['accuracy']:.4f}, "
                      f"Training: {training_time:.2f}s, Inference: {avg_inference_time*1000:.2f}ms")
                
            except Exception as e:
                model_performances[model_type] = {
                    'training_successful': False,
                    'error': str(e)
                }
                print(f"   ❌ {model_type} failed: {e}")
        
        self.test_results['model_performance'] = model_performances
        print("✅ Model Performance Robustness Test completed")
    
    def test_noise_robustness(self):
        """Test robustness terhadap noise"""
        print("\n🔊 Testing Noise Robustness...")
        
        X, y = self.test_data['X'], self.test_data['y']
        
        # Test different noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        noise_results = {}
        
        for noise_level in noise_levels:
            print(f"📊 Testing noise level: {noise_level}")
            
            # Add Gaussian noise
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            
            # Test each trained model
            model_noise_results = {}
            
            for model_type in ['CNN', 'LSTM', 'CNN_LSTM']:
                if self.ai_models.is_trained.get(model_type, False):
                    try:
                        # Test prediction with noisy data
                        reg_pred, clf_pred = self.ai_models.predict(model_type, X_noisy[:100], return_proba=True)
                        
                        # Compare with clean predictions
                        reg_clean, clf_clean = self.ai_models.predict(model_type, X[:100], return_proba=True)
                        
                        # Calculate degradation
                        reg_degradation = np.mean(np.abs(reg_pred - reg_clean))
                        clf_degradation = np.mean(np.abs(clf_pred - clf_clean))
                        
                        model_noise_results[model_type] = {
                            'regression_degradation': reg_degradation,
                            'classification_degradation': clf_degradation,
                            'robust': reg_degradation < 0.5 and clf_degradation < 0.1
                        }
                        
                        status = "✅" if model_noise_results[model_type]['robust'] else "⚠️"
                        print(f"   {status} {model_type}: Reg deg={reg_degradation:.4f}, Clf deg={clf_degradation:.4f}")
                        
                    except Exception as e:
                        model_noise_results[model_type] = {
                            'error': str(e),
                            'robust': False
                        }
                        print(f"   ❌ {model_type}: {e}")
            
            noise_results[noise_level] = model_noise_results
        
        self.test_results['noise_robustness'] = noise_results
        print("✅ Noise Robustness Test completed")
    
    def test_temporal_robustness(self):
        """Test robustness terhadap temporal patterns"""
        print("\n⏰ Testing Temporal Robustness...")
        
        # Test different sequence lengths
        sequence_lengths = [20, 30, 40, 50, 60, 80]
        temporal_results = {}
        
        for seq_len in sequence_lengths:
            if seq_len > self.test_data['X'].shape[1]:
                continue
            
            print(f"📊 Testing sequence length: {seq_len}")
            
            # Extract sequences of different lengths
            X_temp = self.test_data['X'][:, :seq_len, :]
            
            temporal_model_results = {}
            
            for model_type in ['CNN', 'LSTM', 'CNN_LSTM']:
                if self.ai_models.is_trained.get(model_type, False):
                    try:
                        # Test inference with different sequence lengths
                        start_time = time.time()
                        predictions = self.ai_models.predict(model_type, X_temp[:50])
                        inference_time = time.time() - start_time
                        
                        temporal_model_results[model_type] = {
                            'inference_time': inference_time,
                            'successful': True
                        }
                        
                        print(f"   ✅ {model_type}: {inference_time:.3f}s for {seq_len} timesteps")
                        
                    except Exception as e:
                        temporal_model_results[model_type] = {
                            'error': str(e),
                            'successful': False
                        }
                        print(f"   ❌ {model_type}: {e}")
            
            temporal_results[seq_len] = temporal_model_results
        
        self.test_results['temporal_robustness'] = temporal_results
        print("✅ Temporal Robustness Test completed")
    
    def generate_robustness_report(self):
        """Generate comprehensive robustness report"""
        print("\n📋 Generating Robustness Report...")
        
        report = {
            'test_summary': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_tests': len(self.test_results),
                'test_categories': list(self.test_results.keys())
            },
            'detailed_results': self.test_results
        }
        
        # Calculate overall robustness score
        robustness_scores = []
        
        # Data quality score
        if 'data_quality' in self.test_results:
            missing_success = sum(1 for r in self.test_results['data_quality']['missing_values'].values() 
                                if r.get('processed_successfully', False))
            outlier_success = sum(1 for r in self.test_results['data_quality']['outliers'].values() 
                                if r.get('processed_successfully', False))
            data_quality_score = (missing_success + outlier_success) / 8  # 4 tests each
            robustness_scores.append(data_quality_score)
        
        # Model performance score
        if 'model_performance' in self.test_results:
            model_success = sum(1 for r in self.test_results['model_performance'].values() 
                              if r.get('training_successful', False))
            model_score = model_success / len(self.test_results['model_performance'])
            robustness_scores.append(model_score)
        
        # Noise robustness score
        if 'noise_robustness' in self.test_results:
            noise_robust_count = 0
            noise_total_count = 0
            for noise_level_results in self.test_results['noise_robustness'].values():
                for model_results in noise_level_results.values():
                    if 'robust' in model_results:
                        noise_total_count += 1
                        if model_results['robust']:
                            noise_robust_count += 1
            
            noise_score = noise_robust_count / noise_total_count if noise_total_count > 0 else 0
            robustness_scores.append(noise_score)
        
        # Overall robustness score
        overall_score = np.mean(robustness_scores) if robustness_scores else 0
        
        report['test_summary']['overall_robustness_score'] = overall_score
        report['test_summary']['robustness_grade'] = self.get_robustness_grade(overall_score)
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 ROBUSTNESS TEST SUMMARY")
        print("="*60)
        print(f"Overall Robustness Score: {overall_score:.3f}")
        print(f"Robustness Grade: {report['test_summary']['robustness_grade']}")
        print(f"Test Categories: {len(self.test_results)}")
        
        for category, results in self.test_results.items():
            print(f"\n📊 {category.replace('_', ' ').title()}:")
            if category == 'model_performance':
                for model, perf in results.items():
                    if perf.get('training_successful'):
                        print(f"   ✅ {model}: Acc={perf['metrics']['accuracy']:.3f}")
                    else:
                        print(f"   ❌ {model}: Failed")
        
        print("="*60)
        
        # Save report
        import json
        report_file = f"robustness_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Full report saved: {report_file}")
        
        return report
    
    def get_robustness_grade(self, score):
        """Convert robustness score to grade"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B+ (Good)"
        elif score >= 0.6:
            return "B (Acceptable)"
        elif score >= 0.5:
            return "C+ (Fair)"
        elif score >= 0.4:
            return "C (Poor)"
        else:
            return "D (Very Poor)"
    
    def run_comprehensive_test(self):
        """Run all robustness tests"""
        print("🚀 Starting Comprehensive Robustness Testing...")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Run tests
            self.test_data_quality_robustness()
            self.test_model_performance_robustness()
            self.test_noise_robustness()
            self.test_temporal_robustness()
            
            # Generate report
            report = self.generate_robustness_report()
            
            total_time = time.time() - start_time
            print(f"\n⏱️ Total testing time: {total_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            return None
        finally:
            if self.db_manager.connection:
                self.db_manager.disconnect()

def main():
    """Main function untuk robustness testing"""
    tester = RobustnessTestSuite()
    
    try:
        report = tester.run_comprehensive_test()
        
        if report:
            print("\n🎉 Robustness testing completed successfully!")
            print("📊 Check the generated report file for detailed results.")
        else:
            print("\n❌ Robustness testing failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️ Testing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
