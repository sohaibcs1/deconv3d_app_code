1: Create env

python3 -m venv deconv3d

source deconv3d/bin/activate

2: Instaltion

pip install -r requirements.txt
 
3: Run
python3 app.py

+++
List databse
sqlite3 users.db
SELECT name FROM sqlite_master WHERE type='table';
PRAGMA table_info(logs);

SELECT * FROM users;


+++++++++


(Delete old exp)after 10 Time
10 * 60 * 60

now (12:00) - last_seen_ts (10:00) = 2 hours

+++++
When someone request
    *app check the key 
        ---> then process any function request
        ---> reject/clear session

[* data (your session content)+APP_SECRET_KEY]
+++

if 100 users are using at same time
    * only serve 2 jobs at a time (from all users) others will wait [FIFO]
    
MAX_CONCURRENT = int(os.getenv("DECONV3D_MAX_CONCURRENT", "2"))

+++++


Run/kill
chmod +x run.sh
chmod +x kill.sh

./run.sh
./kill.sh

