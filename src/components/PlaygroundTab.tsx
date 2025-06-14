import React, { useState } from 'react';
import { Send, Copy, RefreshCw, Settings, Sparkles } from 'lucide-react';

export const PlaygroundTab: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [settings, setSettings] = useState({
    temperature: 0.7,
    maxLength: 200,
    topP: 0.9,
    topK: 50,
  });

  const samplePrompts = [
    "Once upon a time in a distant galaxy,",
    "The future of artificial intelligence will",
    "In the year 2050, technology will have",
    "The most important lesson I learned was",
  ];

  const handleGenerate = async () => {
    if (!prompt.trim()) return;
    
    setIsGenerating(true);
    // Simulate text generation
    setTimeout(() => {
      setGeneratedText(`${prompt} this is a sample generated continuation of your prompt. The model would generate contextually relevant text based on your fine-tuned dataset. This text demonstrates how the fine-tuned GPT-2 model continues the narrative in a coherent and stylistically consistent manner.`);
      setIsGenerating(false);
    }, 2000);
  };

  return (
    <div className="space-y-8">
      {/* Text Generation Interface */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Text Generation Playground</h3>
        
        <div className="grid lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 space-y-6">
            {/* Prompt Input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Enter your prompt
              </label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Start typing your prompt here..."
                className="w-full h-32 p-4 border rounded-lg resize-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            {/* Sample Prompts */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">
                Or try these sample prompts:
              </label>
              <div className="grid md:grid-cols-2 gap-2">
                {samplePrompts.map((samplePrompt, index) => (
                  <button
                    key={index}
                    onClick={() => setPrompt(samplePrompt)}
                    className="text-left p-3 text-sm border rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors"
                  >
                    {samplePrompt}
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={!prompt.trim() || isGenerating}
              className="flex items-center space-x-2 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isGenerating ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  <span>Generating...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4" />
                  <span>Generate Text</span>
                </>
              )}
            </button>
          </div>

          {/* Generation Settings */}
          <div className="bg-gray-50 p-6 rounded-lg">
            <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
              <Settings className="h-4 w-4 mr-2" />
              Generation Settings
            </h4>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Temperature: {settings.temperature}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) => setSettings({...settings, temperature: parseFloat(e.target.value)})}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">Controls randomness</p>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Length: {settings.maxLength}
                </label>
                <input
                  type="range"
                  min="50"
                  max="500"
                  step="10"
                  value={settings.maxLength}
                  onChange={(e) => setSettings({...settings, maxLength: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Top-p: {settings.topP}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  value={settings.topP}
                  onChange={(e) => setSettings({...settings, topP: parseFloat(e.target.value)})}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">Nucleus sampling</p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Top-k: {settings.topK}
                </label>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={settings.topK}
                  onChange={(e) => setSettings({...settings, topK: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Generated Output */}
      {generatedText && (
        <div className="bg-white rounded-xl shadow-md p-8">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-900">Generated Text</h3>
            <button
              onClick={() => navigator.clipboard.writeText(generatedText)}
              className="flex items-center space-x-2 text-blue-600 hover:text-blue-800"
            >
              <Copy className="h-4 w-4" />
              <span>Copy</span>
            </button>
          </div>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">
              {generatedText}
            </p>
          </div>
        </div>
      )}

      {/* Model Comparison */}
      <div className="bg-white rounded-xl shadow-md p-8">
        <h3 className="text-2xl font-bold text-gray-900 mb-6">Model Comparison</h3>
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
              <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
              Fine-tuned Model
            </h4>
            <div className="bg-blue-50 p-4 rounded-lg">
              <p className="text-gray-800 text-sm">
                Your custom fine-tuned GPT-2 model trained on domain-specific data. 
                This model should demonstrate improved performance on your specific use case.
              </p>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
              <div className="w-3 h-3 bg-gray-400 rounded-full mr-2"></div>
              Base GPT-2
            </h4>
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-gray-800 text-sm">
                The original GPT-2 model without fine-tuning. Use this as a baseline 
                to compare the improvements from your fine-tuning process.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};