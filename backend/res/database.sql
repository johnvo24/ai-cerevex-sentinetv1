DROP TABLE IF EXISTS ArticleTag;
DROP TABLE IF EXISTS Tag;
DROP TABLE IF EXISTS Comment;
DROP TABLE IF EXISTS Article;
DROP TABLE IF EXISTS Users;

CREATE TABLE Users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    email VARCHAR(255),
    role BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Article (
    id SERIAL PRIMARY KEY,
    user_id INT,
    model_id INT,
    title VARCHAR(255),
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES Users(id)
);

CREATE TABLE Comment (
    id SERIAL PRIMARY KEY,
    article_id INT,
    user_id INT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_id) REFERENCES Article(id),
    FOREIGN KEY (user_id) REFERENCES Users(id)
);

CREATE TABLE Tag (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255)
);

CREATE TABLE ArticleTag (
    article_id INT,
    tag_id INT,
    PRIMARY KEY (article_id, tag_id),
    FOREIGN KEY (article_id) REFERENCES Article(id),
    FOREIGN KEY (tag_id) REFERENCES Tag(id)
);
