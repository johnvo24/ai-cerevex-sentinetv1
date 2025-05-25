import { ArticleCard } from "../components/ArticleCard"

const articles = [
  {
    id: 1,
    image: "https://i.imgur.com/nyIo5tK.png",
    title: "Article title 1",
    category: "Sports",
    date: "20 July 2021",
  },
  {
    id: 2,
    image: "https://i.imgur.com/BpHE3Z7.png",
    title: "Article title 2",
    category: "Business",
    date: "20 August 2021",
  },
  {
    id: 3,
    image: "https://i.imgur.com/8nJkYQg.png",
    title: "Article title 3",
    category: "Politics",
    date: "5 September 2021",
  },
  {
    id: 4,
    image: "https://i.imgur.com/8nJkYQg.png",
    title: "Article title 4",
    category: "Politics",
    date: "12 September 2022",
  },
]

export const Home = () => {
  return (
    <div className="min-h-screen px-6 py-8">
      <div className="max-w-5xl mx-auto grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {articles.map((article) => (
          <ArticleCard
            key={article.id}
            image={article.image}
            title={article.title}
            category={article.category}
            date={article.date}
          />
        ))}
      </div> 
    </div>
  )
}