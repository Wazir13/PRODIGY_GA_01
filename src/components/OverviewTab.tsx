import React from 'react';
import { ArrowRight, CheckCircle, Clock, Cpu } from 'lucide-react';

export const OverviewTab: React.FC = () => {
  const steps = [
    {
      title: 'Data Preparation',
      description: 'Collect and preprocess your training dataset',
      status: 'completed',
      icon: CheckCircle,
    },
    {
      title: 'Model Configuration',
      description: 'Set up GPT-2 model parameters and training settings',
      status: 'in-progress',
      icon: Clock,
    },
    {
      title: 'Fine-tuning',
      description: 'Train the model on your custom dataset',
      status: 'pending',
      icon: Cpu,
    },
    {
      title: 'Evaluation',
      description: 'Test and validate model performance',
      status: 'pending',
      icon: CheckCircle,
    },
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-2xl p-8">
        <div className="max-w-3xl">
          <h2 className="text-3xl font-bold mb-4">GPT-2 Fine-tuning Project</h2>
          <p className="text-blue-100 text-lg mb-6">
            Learn to train a transformer model to generate coherent, contextually relevant text
            that mimics your training data's style and structure.
          </p>
          <div className="flex items-center space-x-4">
            <div className="bg-white/20 px-4 py-2 rounded-lg backdrop-blur-sm">
              <span className="text-sm font-medium">Model: GPT-2</span>
            </div>
            <div className="bg-white/20 px-4 py-2 rounded-lg backdrop-blur-sm">
              <span className="text-sm font-medium">Framework: PyTorch</span>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Steps */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
        {steps.map((step, index) => {
          const Icon = step.icon;
          return (
            <div key={index} className="bg-white rounded-xl shadow-md p-6 hover:shadow-lg transition-shadow">
              <div className="flex items-center mb-4">
                <div className={`p-2 rounded-lg ${
                  step.status === 'completed' ? 'bg-green-100 text-green-600' :
                  step.status === 'in-progress' ? 'bg-blue-100 text-blue-600' :
                  'bg-gray-100 text-gray-400'
                }`}>
                  <Icon className="h-5 w-5" />
                </div>
                <span className={`ml-3 px-2 py-1 rounded-full text-xs font-medium ${
                  step.status === 'completed' ? 'bg-green-100 text-green-800' :
                  step.status === 'in-progress' ? 'bg-blue-100 text-blue-800' :
                  'bg-gray-100 text-gray-600'
                }`}>
                  {step.status === 'completed' ? 'Done' : 
                   step.status === 'in-progress' ? 'Active' : 'Pending'}
                </span>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">{step.title}</h3>
              <p className="text-gray-600 text-sm">{step.description}</p>
            </div>
          );
        })}
      </div>

      {/* Key Concepts */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Key Concepts</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">What is GPT-2?</h4>
            <p className="text-gray-600 mb-4">
              GPT-2 (Generative Pre-trained Transformer 2) is a large-scale unsupervised language model 
              that generates coherent paragraphs of text. It uses the transformer architecture and was 
              trained on a diverse dataset of internet text.
            </p>
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="text-blue-800 text-sm">
                <strong>Architecture:</strong> Transformer decoder with multi-head self-attention
              </p>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Fine-tuning Process</h4>
            <p className="text-gray-600 mb-4">
              Fine-tuning adapts a pre-trained model to your specific domain by continuing training 
              on your dataset. This allows the model to learn domain-specific patterns while retaining 
              general language understanding.
            </p>
            <div className="bg-purple-50 p-4 rounded-lg">
              <p className="text-purple-800 text-sm">
                <strong>Benefits:</strong> Better performance, domain adaptation, reduced training time
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};