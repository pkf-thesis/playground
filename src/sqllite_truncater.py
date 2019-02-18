import os
import sys
import sqlite3

def readDatabase(dbpath):
    if not os.path.isfile(dbpath):
        print ('ERROR: db file %s does not exist' % dbpath)
        sys.exit(0)
    return dbpath

def deleteNonMusicRelatedTags(dbfile):
    this_function_name = sys._getframe().f_code.co_name

    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sql = "DELETE FROM tids WHERE tids.ROWID IN (SELECT tid FROM tid_tag WHERE tag IN (SELECT tags.ROWID FROM tags WHERE tag IN ('Favourites', 'Favorite', 'favorites', 'Awesome', 'seen live', 'cool', 'british')))"
    cursor.execute(sql)
    print(str(this_function_name + ": records("+str(cursor.rowcount)+") deleted"))

    sql = "DELETE FROM tid_tag WHERE tag IN (SELECT tags.ROWID FROM tags WHERE tag IN ('Favourites', 'Favorite', 'favorites', 'Awesome', 'seen live', 'cool', 'british'))"
    cursor.execute(sql)
    print(str(this_function_name + ": records("+str(cursor.rowcount)+") deleted"))

    sql = "DELETE FROM tags WHERE tag IN ('Favourites', 'Favorite', 'favorites', 'Awesome', 'seen live', 'cool', 'british')"
    cursor.execute(sql)
    print(str(this_function_name + ": records("+str(cursor.rowcount)+") deleted"))

    conn.commit()
    
    conn.close()

def deleteEverythingButTop50Tags(dbfile):
    this_function_name = sys._getframe().f_code.co_name

    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sql = "DELETE FROM tids WHERE tids.ROWID NOT IN (SELECT tid FROM tid_tag WHERE tag IN (SELECT tag FROM tid_tag GROUP BY tag ORDER BY COUNT(tid) DESC LIMIT 50))"
    cursor.execute(sql)
    print(str(this_function_name + ": records("+str(cursor.rowcount)+") deleted"))

    sql = "DELETE FROM tid_tag WHERE tag NOT IN (SELECT tag FROM tid_tag GROUP BY tag ORDER BY COUNT(tid) DESC LIMIT 50)"
    cursor.execute(sql)
    print(str(this_function_name + ": records("+str(cursor.rowcount)+") deleted"))

    sql = "DELETE FROM tags WHERE tags.ROWID NOT IN (SELECT tag FROM tid_tag GROUP BY tag ORDER BY COUNT(tid) DESC LIMIT 50)"
    cursor.execute(sql)
    print(str(this_function_name + ": records("+str(cursor.rowcount)+") deleted"))

    conn.commit()

    conn.close()

def vacuum(dbfile):
    this_function_name = sys._getframe().f_code.co_name

    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sql = "vacuum"
    cursor.execute(sql)
    print(str(this_function_name + ": done"))

    conn.close()

def createTable(dbfile):
    this_function_name = sys._getframe().f_code.co_name
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()

    sql = "CREATE TABLE IF NOT EXISTS meta_data (track_id Varchar, title Varchar, song_id Varchar, release Varchar, artist_id Varchar, artist_name Varchar, duration Varchar, year int)"
    cursor.execute(sql)
    print(str(this_function_name + ": done"))

    conn.close()

def joinMetaData(lastfm, metadata):
    this_function_name = sys._getframe().f_code.co_name

    conn = sqlite3.connect(lastfm)
    cursor = conn.cursor()

    sql = "ATTACH database '"+metadata+"' as metadatadb"
    cursor.execute(sql)
    sql = "INSERT INTO meta_data SELECT track_id, title, song_id, release, artist_id, artist_name, duration, year FROM metadatadb.songs, main.tids WHERE track_id = tids.tid"
    cursor.execute(sql)

    print(str(this_function_name + ": done"))

    conn.commit()
    conn.close()

# Main
lastfm = readDatabase("../db/lastfm_tags.db")
metadata = "../db/track_metadata.db"

deleteNonMusicRelatedTags(lastfm)
deleteEverythingButTop50Tags(lastfm)
vacuum(lastfm)
createTable(lastfm)
joinMetaData(lastfm, metadata)