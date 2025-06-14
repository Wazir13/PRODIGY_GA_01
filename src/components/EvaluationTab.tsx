import React from 'react';
import { TrendingUp, Target, Award, BarChart3 } from 'lucide-react';

export const EvaluationTab: React.FC = () => {
  const metrics = [
    { name: 'Perplexity', value: 23.45, improvement: -12.3, color: 'blue' },
    { name: 'BLEU Score', value: 0.68, improvement: 15.2, color: 'green' },
    { name: 'Coherence', value: 0.84, improvement: 8.7, color: 'purple' },
    { name: 'Fluency', value: 0.92, improvement: 6.1, color: 'orange' },
  ];

  const trainingCurve = [
    { epoch: 1, trainLoss: 2.34, valLoss: 2.41 },
    { epoch: 2, trainLoss: 1.87, valLoss: 1.95 },
    { epoch: 3, trainLoss: 1.52, valLoss: 1.68 },
  ];

  return (
    <div className="space-y-8">
      {/* Performance Metrics */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Model Performance Metrics</h3>
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {metrics.map((metric, index) => (
            <div key={index} className={`p-6 rounded-lg bg-${metric.color}-50`}>
              <div className="flex items-center justify-between mb-2">
                <Target className={`h-6 w-6 text-${metric.color}-600`} />
                <span className={`text-xs px-2 py-1 rounded-full ${
                  metric.improvement > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {metric.improvement > 0 ? '+' : ''}{metric.improvement.toFixed(1)}%
                </span>
              </div>
              <h4 className={`text-2xl font-bold text-${metric.color}-900 mb-1`}>
                {metric.value}
              </h4>
              <p className={`text-${metric.color}-700 text-sm`}>{metric.name}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Training Curves */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Training Progress</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Loss Curves</h4>
            <div className="bg-gray-50 p-6 rounded-lg">
              <div className="space-y-4">
                {trainingCurve.map((point, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Epoch {point.epoch}</span>
                    <div className="flex space-x-4">
                      <span className="text-sm">
                        <span className="text-blue-600">Train:</span> {point.trainLoss}
                      </span>
                      <span className="text-sm">
                        <span className="text-purple-600">Val:</span> {point.valLoss}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Key Insights</h4>
            <div className="space-y-3">
              <div className="p-3 bg-green-50 rounded-lg border-l-4 border-green-500">
                <p className="text-green-800 text-sm">
                  <strong>Good convergence:</strong> Loss is decreasing steadily across epochs
                </p>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg border-l-4 border-blue-500">
                <p className="text-blue-800 text-sm">
                  <strong>No overfitting:</strong> Validation loss follows training loss closely
                </p>
              </div>
              <div className="p-3 bg-yellow-50 rounded-lg border-l-4 border-yellow-500">
                <p className="text-yellow-800 text-sm">
                  <strong>Recommendation:</strong> Consider training for 1-2 more epochs
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sample Generations */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Sample Generations</h3>
        <div className="space-y-6">
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Prompt: "The future of technology will"</h4>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Fine-tuned Model</h5>
                <div className="bg-blue-50 p-4 rounded-lg">
                  <p className="text-gray-800 text-sm">
                    The future of technology will fundamentally reshape how we interact with 
                    digital systems, creating more intuitive and accessible interfaces that 
                    bridge the gap between human cognition and machine capabilities.
                  </p>
                </div>
                <div className="mt-2 flex items-center space-x-2">
                  <Award className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-green-600">Quality Score: 8.7/10</span>
                </div>
              </div>
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-2">Base GPT-2</h5>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="text-gray-800 text-sm">
                    The future of technology will be a lot of fun and I think it's going to be 
                    a great way to get people to think about what they want to do with their lives.
                  </p>
                </div>
                <div className="mt-2 flex items-center space-x-2">
                  <Award className="h-4 w-4 text-orange-600" />
                  <span className="text-sm text-orange-600">Quality Score: 6.2/10</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Evaluation Summary */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-xl p-8">
        <div className="flex items-center mb-4">
          <BarChart3 className="h-8 w-8 mr-3" />
          <h3 className="text-2xl font-bold">Evaluation Summary</h3>
        </div>
        <div className="grid md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-semibold mb-2">Overall Performance</h4>
            <p className="text-green-100 text-sm">
              Your fine-tuned model shows significant improvement over the base GPT-2, 
              with better coherence and domain-specific knowledge.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-2">Key Strengths</h4>
            <ul className="text-green-100 text-sm space-y-1">
              <li>• Improved domain relevance</li>
              <li>• Better contextual understanding</li>
              <li>• More coherent long-form text</li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-2">Next Steps</h4>
            <ul className="text-green-100 text-sm space-y-1">
              <li>• Deploy to production</li>
              <li>• Collect user feedback</li>
              <li>• Plan iterative improvements</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};