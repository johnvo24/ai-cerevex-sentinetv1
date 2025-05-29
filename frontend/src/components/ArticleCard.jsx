import { Link } from "react-router-dom"

export const ArticleCard = ({ image, title, category, date }) => {
  return (
    <div className="max-w-xs rounded-3xl overflow-hidden bg-[#f8f3ea] border-1 border-[#2625223D] shadow-md">
      <img
        src={image}
        alt={title}
        className="w-full h-48 object-cover rounded-tl-lg rounded-tr-lg"
      />
      <div className="p-3">
        <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
        <span className="inline-block mt-2 text-xs bg-red-200 text-red-700 px-3 py-1 rounded-full uppercase tracking-wide">
          {category}
        </span>
        <div className="flex justify-between mt-4">
          <p className="text-xs text-gray-500 mt-2">{date}</p>
          <Link
            to="/view-article"
            className="text-sm px-4 py-1 border border-gray-400 rounded-full hover:bg-gray-100"
          >
            View
          </Link>
        </div>
      </div>
    </div>
  )
}
