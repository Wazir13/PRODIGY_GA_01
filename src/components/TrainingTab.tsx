import React, { useState } from 'react';
import { Play, Pause, Settings, TrendingUp, Clock, Cpu, MemoryStick } from 'lucide-react';

export const TrainingTab: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress] = useState(68);

  const [trainingConfig, setTrainingConfig] = useState({
    learningRate: 5e-5,
    batchSize: 16,
    epochs: 3,
    warmupSteps: 500,
    maxLength: 512,
  });

  return (
    <div className="space-y-8">
      {/* Training Configuration */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Training Configuration</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Model Parameters</h4>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Learning Rate
                </label>
                <input 
                  type="number" 
                  step="0.00001"
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500" 
                  value={trainingConfig.learningRate}
                  onChange={(e) => setTrainingConfig({...trainingConfig, learningRate: parseFloat(e.target.value)})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Batch Size
                </label>
                <select 
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500"
                  value={trainingConfig.batchSize}
                  onChange={(e) => setTrainingConfig({...trainingConfig, batchSize: parseInt(e.target.value)})}
                >
                  <option value={8}>8</option>
                  <option value={16}>16</option>
                  <option value={32}>32</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Epochs
                </label>
                <input 
                  type="number" 
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500" 
                  value={trainingConfig.epochs}
                  onChange={(e) => setTrainingConfig({...trainingConfig, epochs: parseInt(e.target.value)})}
                />
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Advanced Settings</h4>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Warmup Steps
                </label>
                <input 
                  type="number" 
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500" 
                  value={trainingConfig.warmupSteps}
                  onChange={(e) => setTrainingConfig({...trainingConfig, warmupSteps: parseInt(e.target.value)})}
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Sequence Length
                </label>
                <input 
                  type="number" 
                  className="w-full p-3 border rounded-lg focus:ring-2 focus:ring-blue-500" 
                  value={trainingConfig.maxLength}
                  onChange={(e) => setTrainingConfig({...trainingConfig, maxLength: parseInt(e.target.value)})}
                />
              </div>
              <div>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-3" />
                  <span className="text-gray-700">Enable gradient accumulation</span>
                </label>
              </div>
              <div>
                <label className="flex items-center">
                  <input type="checkbox" className="mr-3" defaultChecked />
                  <span className="text-gray-700">Save checkpoints</span>
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Control */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-2xl font-bold text-gray-900">Training Control</h3>
          <div className="flex space-x-3">
            <button 
              onClick={() => setIsTraining(!isTraining)}
              className={`flex items-center space-x-2 px-6 py-2 rounded-lg font-medium transition-colors ${
                isTraining 
                  ? 'bg-red-600 text-white hover:bg-red-700' 
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {isTraining ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              <span>{isTraining ? 'Pause Training' : 'Start Training'}</span>
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 border rounded-lg hover:bg-gray-50">
              <Settings className="h-4 w-4" />
              <span>Advanced</span>
            </button>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700">Training Progress</span>
            <span className="text-sm text-gray-500">{trainingProgress}% Complete</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-300"
              style={{ width: `${trainingProgress}%` }}
            ></div>
          </div>
        </div>

        {/* Training Metrics */}
        <div className="grid md:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-blue-50 rounded-lg">
            <Clock className="h-6 w-6 text-blue-600 mx-auto mb-2" />
            <h4 className="text-lg font-bold text-blue-900">2h 34m</h4>
            <p className="text-blue-700 text-sm">Time Elapsed</p>
          </div>
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <TrendingUp className="h-6 w-6 text-green-600 mx-auto mb-2" />
            <h4 className="text-lg font-bold text-green-900">1.24</h4>
            <p className="text-green-700 text-sm">Current Loss</p>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <Cpu className="h-6 w-6 text-purple-600 mx-auto mb-2" />
            <h4 className="text-lg font-bold text-purple-900">85%</h4>
            <p className="text-purple-700 text-sm">GPU Usage</p>
          </div>
          <div className="text-center p-4 bg-orange-50 rounded-lg">
            <MemoryStick className="h-6 w-6 text-orange-600 mx-auto mb-2" />
            <h4 className="text-lg font-bold text-orange-900">12.4 GB</h4>
            <p className="text-orange-700 text-sm">Memory Used</p>
          </div>
        </div>
      </div>

      {/* Training Logs */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Training Logs</h3>
        <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-64 overflow-y-auto">
          <div>[2024-01-15 14:23:45] Starting training...</div>
          <div>[2024-01-15 14:23:46] Loading model checkpoint...</div>
          <div>[2024-01-15 14:23:48] Initializing optimizer...</div>
          <div>[2024-01-15 14:23:50] Training started</div>
          <div>[2024-01-15 14:24:12] Epoch 1/3 - Step 100/1500 - Loss: 2.34</div>
          <div>[2024-01-15 14:24:35] Epoch 1/3 - Step 200/1500 - Loss: 2.12</div>
          <div>[2024-01-15 14:24:58] Epoch 1/3 - Step 300/1500 - Loss: 1.98</div>
          <div>[2024-01-15 14:25:21] Epoch 1/3 - Step 400/1500 - Loss: 1.87</div>
          <div>[2024-01-15 14:25:44] Checkpoint saved at step 500</div>
          <div>[2024-01-15 14:26:07] Epoch 1/3 - Step 500/1500 - Loss: 1.76</div>
          <div className="text-yellow-400">[2024-01-15 14:26:30] Current training step: 680/1500</div>
        </div>
      </div>
    </div>
  );
};