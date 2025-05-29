import { BrowserRouter, Route, Routes } from "react-router-dom"
import { Home } from "./pages/Home"
import { CreateArticle } from "./pages/CreateArticle"
import { ArticleDetail } from "./pages/ArticleDetail"
import { Navbar } from "./components/Navbar"

function App() {
  return (
    <>
      <BrowserRouter>
        <div className="px-10 py-4">
          <Navbar />
          <Routes>
            <Route index element={<Home/>}/>
            <Route path="/create-article" element={<CreateArticle/>}/>
            <Route path="/view-article" element={<ArticleDetail/>}/>
          </Routes>
        </div>
      </BrowserRouter>
    </>
  )
}

export default App
