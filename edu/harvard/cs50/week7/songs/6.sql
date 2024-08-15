--
-- Write a query that lists the names of songs that are by Post Malone
--

SELECT songs.name FROM songs join artists on songs.artist_id = artists.id where artists.name = 'Post Malone';