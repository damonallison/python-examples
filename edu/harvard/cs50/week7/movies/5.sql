-- list the titles and release years of all Harry Potter movies, in chronological order

select title, year from movies where title like '%harry potter%' order by year asc;