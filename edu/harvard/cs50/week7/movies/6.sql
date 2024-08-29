-- determine the average rating of all movies released in 2012

select avg(rating) from ratings as r join movies as m on r.movie_id = m.id where m.year = 2012;