import { NavLink } from "react-router-dom"
import { Search } from "lucide-react"

export const Navbar = () => {
  const linkClass = ({ isActive }) => isActive ? "font-bold text-black border-b-2 border-[#EE6352]" : ""

  return (
    <header className="flex justify-between items-center px-6 py-4 bg-[#f8f3ea] border-1 border-[#2625223D] rounded-full shadow-md shadow-[#2625223D]-500/50">
      <div className="text-xl font-semibold">Logo</div>
      <nav className="hidden md:flex gap-8 uppercase text-sm text-gray-600">
        <NavLink to="/" className={linkClass}>Home</NavLink>
        <NavLink to="/create-article" className={linkClass}>Create Article</NavLink>
        <a href="#">Page 2</a>
        <a href="#">Page 3</a>
      </nav>
      <div className="flex items-center gap-4">
        <button className="p-2 rounded-2xl bg-[#26252214] shadow">
          <Search />
        </button>
        <button className="px-4 py-2 bg-black text-white text-sm rounded-full cursor-pointer hover:opacity-75">Sign Up</button>
      </div>
    </header>
  )
}