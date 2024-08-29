-- list all movies released in 2010 and their ratings, in descending order by rating.
-- for movies with the same rating, order them alphabetically by title

select m.title, r.rating from movies as m join ratings as r on m.id = r.movie_id where m.year = 2010 order by rating desc, title asc;