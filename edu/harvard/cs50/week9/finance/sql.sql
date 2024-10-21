DROP TABLE IF EXISTS transactions;

CREATE TABLE transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    user_id INT NOT NULL,
    tx_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    shares INT NOT NULL,
    price NUMERIC NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

insert into transactions (user_id, tx_type, symbol, shares, price) values (1, 'buy', 'msft', 100, 12.34);

