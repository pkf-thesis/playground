import os
import sys
import sqlite3

# Example usage
#print (fetchTagFromSong("TRCCCYE12903CFF0E9"))
#songTags = fetchTagsFromSongs(["TRCCCYE12903CFF0E9", "TRAAABD128F429CF47"])
#print (songTags.get("TRAAABD128F429CF47"))

def database():
    dbfile = "../db/lastfm_tags.db"

    if not os.path.isfile(dbfile):
        print ('ERROR: db file %s does not exist' % dbfile)
        sys.exit(0)
    return dbfile

def deleteEverythingButTop50Tags():
    dbfile = database()
    conn = sqlite3.connect(dbfile)

    sql = "SELECT tag, COUNT(tid) AS countTid FROM tid_tag GROUP BY tag ORDER BY countTid DESC LIMIT 50"
    res = conn.execute(sql)
    data = res.fetchall()

    print(data)

    conn.close()

# Main
database()
deleteEverythingButTop50Tags()

# DELETE 
# FROM dbo.ErrorLog
# WHERE  
# ErrorLogId NOT IN ( SELECT TOP ( 2 )
#                             ErrorLogId
#                     FROM    dbo.ErrorLog
#                     ORDER BY ErrorDate )