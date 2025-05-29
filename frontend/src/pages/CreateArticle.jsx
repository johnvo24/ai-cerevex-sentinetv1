import { useState } from "react";

export const CreateArticle = () => {
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Article submitted:", { title, content });
    alert("Bài báo đã được gửi!");
  };

  return (
    <div className="max-w-3xl mx-auto p-6 bg-[#f8f3ea] border-1 border-[#2625223D] shadow-md rounded-[32px] mt-6">
      <h2 className="text-3xl font-bold mb-4">Create New Article</h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium">Title</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="w-full mt-1 p-2 border border-gray-300 rounded-md bg-white"
            placeholder="Enter title"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium">Content</label>
          <textarea
            rows="6"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="w-full mt-1 p-2 border border-gray-300 rounded-md bg-white"
            placeholder="Enter content"
            required
          />
        </div>
        <button
          type="submit"
          className="bg-black text-white px-4 py-2 rounded-full hover:opacity-75 cursor-pointer"
        >
          Create Article
        </button>
      </form>
    </div>
  )
}
