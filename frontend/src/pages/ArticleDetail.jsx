import { useState } from "react"
import { Clock12 } from "lucide-react"

export const ArticleDetail = () => {
  const [showFeedback, setShowFeedback] = useState(false);
  const [feedback, setFeedback] = useState("");

  const handleToggleFeedback = () => {
    setShowFeedback((prev) => !prev)
  }

  const handleSubmitFeedback = () => {
    alert(`Feedback đã gửi: ${feedback}`)
    setFeedback("")
    setShowFeedback(false)
  }

  return (
    <div className="min-h-screen px-6 py-8">
      <div className="max-w-5xl mx-auto bg-[#f8f3ea] rounded-[32px] border-1 border-[#2625223D] shadow-md p-6">
        <div className="text-center mb-4">
          <span className="bg-red-500 text-white text-xs px-3 py-1 rounded-full uppercase tracking-wide">
            Sports
          </span>
          <h1 className="text-3xl md:text-4xl font-bold mt-4 uppercase">
            Đội tuyển quốc gia Việt Nam giành chức vô địch
          </h1>
          <p className="flex items-center justify-center text-sm text-gray-500 mt-2">
            <Clock12 className="me-2"/> 20 July 2021
          </p>
        </div>

        <img
          src="https://i.imgur.com/HvErS5v.png"
          alt="Football Team"
          className="w-full rounded-lg object-cover my-4"
        />

        <div className="space-y-4 text-gray-700 leading-relaxed text-sm">
          <p>
            Picture succulent chicken infused with the bright notes of lemon
            and the aromatic richness of garlic...
          </p>
          <p>
            As you preheat your oven, envision the kitchen filling with the
            tantalizing aromas...
          </p>
          <p>
            This recipe goes beyond the basics, inviting you to savor the
            richness...
          </p>
        </div>

        <div className="flex items-center gap-4 mt-8 border-t pt-6">
          <img
            src="https://randomuser.me/api/portraits/women/44.jpg"
            alt="Author"
            className="w-12 h-12 rounded-full"
          />
          <div>
            <h4 className="font-semibold text-sm">Isabella Russo</h4>
            <p className="text-xs text-gray-500">
              A writer with a pinch of zest and passion...
            </p>
          </div>
        </div>

        <div className="flex flex-col md:flex-row items-center justify-between mt-6">
          <button className="mt-4 md:mt-0 bg-white border text-sm border-gray-300 px-6 py-2 rounded-full hover:bg-gray-100">
            Read More
          </button>
          <button 
            className="mt-4 md:mt-0 bg-black text-white text-sm px-6 py-2 rounded-full hover:opacity-75"
            onClick={handleToggleFeedback}
          >
            Feedback
          </button>
        </div>

        {showFeedback && (
          <div className="mt-6">
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              rows={4}
              placeholder="Enter your feedback..."
              className="w-full bg-white border border-gray-300 rounded-xl p-3 text-sm focus:ring-black"
            />
            <button
              onClick={handleSubmitFeedback}
              className="mt-2 bg-blue-700 text-white text-sm px-4 py-2 rounded-full hover:opacity-75 cursor-pointer"
            >
              Submit
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
