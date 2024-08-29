-- list the names of all people who have directed a movie that received a rating of at least 9.0

select distinct(p.name) from people as p join movies as m join directors as d join ratings as r on
    p.id = d.person_id and
    m.id = d.movie_id  and
    m.id = r.movie_id
where r.rating >= 9.0;