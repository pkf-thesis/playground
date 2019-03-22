import os
import sys
import sqlite3

# Example usage
#print (fetchTagFromSong("TRCCCYE12903CFF0E9"))
#songTags = fetchTagsFromSongs(["TRCCCYE12903CFF0E9", "TRAAABD128F429CF47"])
#print (songTags.get("TRAAABD128F429CF47"))


def database():
    db_file = "../../db/lastfm_tags.db"

    if not os.path.isfile(db_file):
        print ('ERROR: db file %s does not exist' % db_file)
        sys.exit(0)
    return db_file


def fetch_tag_from_song(tid):
    db_file = database()
    conn = sqlite3.connect(db_file)

    sql = "SELECT tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND " \
          "tid_tag.tid=tids.ROWID and tids.tid='%s'" % tid
    res = conn.execute(sql)
    data = res.fetchall()

    conn.close()
    return data


def fetch_tags_from_songs(tids):
    tags = {}
    db_file = database()
    conn = sqlite3.connect(db_file)

    str_representation = "','".join(tids)
    sql = "SELECT tids.tid, tags.tag, tid_tag.val FROM tid_tag, tids, tags WHERE tags.ROWID=tid_tag.tag AND " \
          "tid_tag.tid=tids.ROWID and tids.tid IN ('%s')" % str_representation
    res = conn.execute(sql)
    data = res.fetchall()

    conn.close()

    for line in data:
        if line[0] in tags:
            tags[line[0]].append([line[1], line[2]])
        else:
            tags[line[0]] = [[line[1], line[2]]]

    return tags


def fetch_all_songs():
    songs = []

    db_file = database()
    conn = sqlite3.connect(db_file)

    sql = "SELECT * FROM tids"
    res = conn.execute(sql)
    data = res.fetchall()

    for song in data:
        songs.append(song[1])

    return songs
