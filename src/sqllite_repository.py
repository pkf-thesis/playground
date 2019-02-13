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

def fetchTagFromSong(tid):
    dbfile = database()
    conn = sqlite3.connect(dbfile)

    sql = "SELECT tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID and tids.tid='%s'" % tid
    res = conn.execute(sql)
    data = res.fetchall()

    conn.close()
    return data

def fetchTagsFromSongs(tids):
    dict = {}
    dbfile = database()
    conn = sqlite3.connect(dbfile)

    strRepresentation = "','".join(tids)
    sql = "SELECT tids.tid, tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND tid_tag.tid=tids.ROWID and tids.tid IN ('%s')" % strRepresentation
    res = conn.execute(sql)
    data = res.fetchall()

    conn.close()

    for line in data:
        if line[0] in dict:
            dict[line[0]].append([line[1], line[2]])
        else:
            dict[line[0]] = [[line[1], line[2]]]

    return dict