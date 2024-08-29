-- determine the number of movies with an IMDb rating of 10.0

select count(*) from movies as m join ratings as r on m.id = r.movie_id where r.rating = 10.0;