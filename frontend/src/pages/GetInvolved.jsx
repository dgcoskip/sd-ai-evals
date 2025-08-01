import { Link } from 'react-router-dom';

function GetInvolved() {
  return (
    <div className="get-involved-page py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl sm:text-5xl font-bold text-gray-800 mb-6">
            Get Involved
          </h1>
          <p className="text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            We're a diverse community of academics, industry experts, software vendors and individuals interested in guiding the development of the next generation of software for the field. Join us!
          </p>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-green-50 p-8 sm:p-10 rounded-xl">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 sm:gap-10">
            <div className="space-y-8">
              <div className="bg-white p-8 rounded-lg shadow-sm">
                <h3 className="text-xl font-bold mb-6 text-gray-800">
                  Join the Community 
                </h3>
                <ul className="space-y-4 text-gray-600">
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-3 mt-1 text-lg">•</span>
                    <span className="leading-relaxed">
                      We'd love to hear your ideas for new types of engines, engine implementations, or evaluations
                    </span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-3 mt-1 text-lg">•</span>
                    <span className="leading-relaxed">
                      Follow discussions on our mailing list and learn about our parent organization BEAMS run by Institute for Artificial Intelligence and Data Science at University at Buffalo
                    </span>
                  </li>
                </ul>
                
                {/* Community CTA Buttons */}
                <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <a 
                    href="https://groups.io/g/sd-ai/"
                    className="inline-flex items-center justify-center px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg shadow-md hover:shadow-lg"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Mailing List
                  </a>
                  <a 
                    href="https://www.buffalo.edu/ai-data-science/research/beams.html"
                    className="inline-flex items-center justify-center px-6 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg shadow-md hover:shadow-lg"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    BEAMS
                  </a>
                </div>
              </div>
            </div>
            
            <div className="space-y-8">
              <div className="bg-white p-8 rounded-lg shadow-sm">
                <h3 className="text-xl font-bold mb-6 text-gray-800">
                  Developers & Product Owners
                </h3>
                <ul className="space-y-4 text-gray-600">
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-3 mt-1 text-lg">•</span>
                    <span className="leading-relaxed">Help refine existing engines or contribute new AI engines</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-3 mt-1 text-lg">•</span>
                    <span className="leading-relaxed">
                      Improve evaluations used to measure model performance
                    </span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-600 mr-3 mt-1 text-lg">•</span>
                    <span className="leading-relaxed">Integrate SD-AI into your application with our simple API</span>
                  </li>
                </ul>
                
                {/* GitHub CTA Button */}
                <div className="mt-8">
                  <a 
                    href="https://github.com/UB-IAD/sd-ai"
                    className="inline-flex items-center justify-center w-full px-6 py-3 bg-gray-900 hover:bg-gray-800 text-white font-semibold rounded-lg shadow-md hover:shadow-lg"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <svg className="w-5 h-5 mr-3" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      <path fillRule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clipRule="evenodd"></path>
                    </svg>
                    View on GitHub
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default GetInvolved;
