-- list the titles of the five highest movies (in order) that Chadwick Boseman starred in, starting with the highest rated

select m.title from movies as m join people as p join stars as s join ratings as r on s.person_id = p.id and s.movie_id = m.id and r.movie_id = m.id where p.name = 'Chadwick Boseman' order by r.rating desc limit 5;