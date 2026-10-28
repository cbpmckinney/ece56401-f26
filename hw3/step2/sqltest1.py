import sqlite3
conn = sqlite3.connect(":memory:")
cur = conn.cursor()
cur.execute("CREATE TABLE logins(uuid, uid)")
cur.execute("INSERT INTO logins VALUES ('admin', 1)")


query = "SELECT * FROM logins WHERE uuid='{}'".format("asdf'OR'1'='1")
print(query)          # see exactly what the final SQL text looks like
cur.execute(query)
print(cur.fetchone())