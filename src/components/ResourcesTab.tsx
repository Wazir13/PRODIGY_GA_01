import React from 'react';
import { Book, ExternalLink, FileText, Video, Code, Users } from 'lucide-react';

export const ResourcesTab: React.FC = () => {
  const resources = [
    {
      category: 'Documentation',
      icon: Book,
      items: [
        { title: 'GPT-2 Paper', description: 'Language Models are Unsupervised Multitask Learners', type: 'paper', url: '#' },
        { title: 'Transformers Documentation', description: 'Hugging Face Transformers Library', type: 'docs', url: '#' },
        { title: 'Fine-tuning Guide', description: 'Complete guide to fine-tuning language models', type: 'guide', url: '#' },
      ]
    },
    {
      category: 'Code Examples',
      icon: Code,
      items: [
        { title: 'GPT-2 Fine-tuning Script', description: 'Python script for fine-tuning GPT-2', type: 'code', url: '#' },
        { title: 'Data Preprocessing', description: 'Text preprocessing utilities', type: 'code', url: '#' },
        { title: 'Evaluation Metrics', description: 'Scripts for model evaluation', type: 'code', url: '#' },
      ]
    },
    {
      category: 'Learning Materials',
      icon: Video,
      items: [
        { title: 'Transformer Architecture', description: 'Deep dive into transformer models', type: 'video', url: '#' },
        { title: 'Fine-tuning Best Practices', description: 'Tips for successful fine-tuning', type: 'article', url: '#' },
        { title: 'Transfer Learning', description: 'Understanding transfer learning in NLP', type: 'tutorial', url: '#' },
      ]
    },
  ];

  const tips = [
    {
      title: 'Data Quality Matters',
      description: 'Clean, well-formatted data is crucial for successful fine-tuning. Remove noise and ensure consistency.',
    },
    {
      title: 'Start with Small Learning Rates',
      description: 'Use learning rates around 5e-5 to 2e-5 to avoid catastrophic forgetting of pre-trained knowledge.',
    },
    {
      title: 'Monitor for Overfitting',
      description: 'Keep track of validation loss and stop training when it starts increasing.',
    },
    {
      title: 'Experiment with Hyperparameters',
      description: 'Try different batch sizes, sequence lengths, and training epochs to optimize performance.',
    },
  ];

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'paper': return FileText;
      case 'docs': return Book;
      case 'guide': return Book;
      case 'code': return Code;
      case 'video': return Video;
      case 'article': return FileText;
      case 'tutorial': return Users;
      default: return FileText;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'paper': return 'text-blue-600';
      case 'docs': return 'text-green-600';
      case 'guide': return 'text-purple-600';
      case 'code': return 'text-orange-600';
      case 'video': return 'text-red-600';
      case 'article': return 'text-indigo-600';
      case 'tutorial': return 'text-pink-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="space-y-8">
      {/* Quick Tips */}
      <div className="bg-gradient-to-r from-purple-500 to-indigo-500 text-white rounded-xl p-8">
        <h3 className="text-2xl font-bold mb-6">Fine-tuning Best Practices</h3>
        <div className="grid md:grid-cols-2 gap-6">
          {tips.map((tip, index) => (
            <div key={index} className="bg-white/10 p-4 rounded-lg backdrop-blur-sm">
              <h4 className="font-semibold mb-2">{tip.title}</h4>
              <p className="text-purple-100 text-sm">{tip.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Resources */}
      {resources.map((category, categoryIndex) => {
        const CategoryIcon = category.icon;
        return (
          <div key={categoryIndex} className="bg-white rounded-xl shadow-md p-8">
            <div className="flex items-center mb-6">
              <CategoryIcon className="h-6 w-6 text-blue-600 mr-3" />
              <h3 className="text-2xl font-bold text-gray-900">{category.category}</h3>
            </div>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {category.items.map((item, itemIndex) => {
                const TypeIcon = getTypeIcon(item.type);
                return (
                  <div key={itemIndex} className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between mb-3">
                      <TypeIcon className={`h-5 w-5 ${getTypeColor(item.type)}`} />
                      <ExternalLink className="h-4 w-4 text-gray-400" />
                    </div>
                    <h4 className="font-semibold text-gray-900 mb-2">{item.title}</h4>
                    <p className="text-gray-600 text-sm mb-4">{item.description}</p>
                    <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                      View Resource →
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}

      {/* Technical Specifications */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Technical Specifications</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">GPT-2 Model Variants</h4>
            <div className="space-y-3">
              <div className="p-4 border rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">GPT-2 Small</span>
                  <span className="text-sm text-gray-500">117M parameters</span>
                </div>
                <p className="text-sm text-gray-600">Good for experimentation and quick prototyping</p>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">GPT-2 Medium</span>
                  <span className="text-sm text-gray-500">345M parameters</span>
                </div>
                <p className="text-sm text-gray-600">Balanced performance and computational requirements</p>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">GPT-2 Large</span>
                  <span className="text-sm text-gray-500">762M parameters</span>
                </div>
                <p className="text-sm text-gray-600">Higher quality output, requires more resources</p>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-4">Hardware Requirements</h4>
            <div className="space-y-4">
              <div>
                <h5 className="font-medium text-gray-800 mb-2">Minimum Requirements</h5>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• 8GB GPU memory (GTX 1080 Ti or better)</li>
                  <li>• 16GB system RAM</li>
                  <li>• 50GB free disk space</li>
                </ul>
              </div>
              <div>
                <h5 className="font-medium text-gray-800 mb-2">Recommended</h5>
                <ul className="text-sm text-gray-600 space-y-1">
                  <li>• 16GB+ GPU memory (RTX 3080 or better)</li>
                  <li>• 32GB system RAM</li>
                  <li>• SSD storage for faster data loading</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Community and Support */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Community & Support</h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center p-6 border rounded-lg">
            <Users className="h-8 w-8 text-blue-600 mx-auto mb-3" />
            <h4 className="font-semibold text-gray-900 mb-2">Community Forum</h4>
            <p className="text-gray-600 text-sm mb-4">
              Connect with other researchers and practitioners
            </p>
            <button className="text-blue-600 hover:text-blue-800 font-medium">
              Join Discussion
            </button>
          </div>
          <div className="text-center p-6 border rounded-lg">
            <Code className="h-8 w-8 text-green-600 mx-auto mb-3" />
            <h4 className="font-semibold text-gray-900 mb-2">GitHub Repository</h4>
            <p className="text-gray-600 text-sm mb-4">
              Access code examples and contribute to the project
            </p>
            <button className="text-green-600 hover:text-green-800 font-medium">
              View Code
            </button>
          </div>
          <div className="text-center p-6 border rounded-lg">
            <FileText className="h-8 w-8 text-purple-600 mx-auto mb-3" />
            <h4 className="font-semibold text-gray-900 mb-2">Documentation</h4>
            <p className="text-gray-600 text-sm mb-4">
              Comprehensive guides and API documentation
            </p>
            <button className="text-purple-600 hover:text-purple-800 font-medium">
              Read Docs
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};