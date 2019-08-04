import datetime
import sys
import lzma
import struct
import itertools
import threading
import queue
import functools

'''
>>> import zlog
>>> import queue

'''

def split(lines, encoding='utf8'):
    ''' Return an iterator of (date, text) tuples from
        the log lines.

        >>> lines = [
        ...     '2019-07-13T01:17:03.034216+00:00 host text e637-39f180263185\n',
        ...     '2019-07-13T01:17:03.041246+00:00 host text 0263185\n',
        ...     '2019-07-13T01:17:06.041452+00:00 host text 3185\n',
        ...     '2019-07-13T01:19:03.042579+00:00 host text (queue size 0)\n',
        ... ]

        >>> tuples = zlog.split(lines)
        >>> dates, texts = zip(*tuples)
        >>> [d.isoformat() for d in dates]
        ['2019-07-13T01:17:03.034216+00:00',
         '2019-07-13T01:17:03.041246+00:00',
         '2019-07-13T01:17:06.041452+00:00',
         '2019-07-13T01:19:03.042579+00:00']
        >>> texts
        ('host text e637-39f180263185\n',
         'host text 0263185\n',
         'host text 3185\n',
         'host text (queue size 0)\n')
        '''
    for line in lines:
        # TODO: how to find the location of the date time?
        # it could be at the begin or in the middle of the line
        # and we cannot use ' ' as a field separator because the
        # date may have it inside:
        #   2019-07-13T01:17:03.034216+00:00 host text
        #   2.11.59.4 - - [13/Mar/2005:04:05:47 -0500] "POST /fp30reg.dll"
        #   Mar 13 04:05:02 combo syslogd 1.4.1: restart.
        idx = line.find(' ')

        date_string = line[:idx]
        text = line[idx+1:]

        # TODO: we could use 'maya' to parse different formats
        # automatically but not all the formats are supported.
        # For now we "hardcode" the format and using strptime.
        #
        # The other thing is that the parsing will fail if there are any
        # extraneous prefix/suffix which forces us to find the exact date
        # from the log line before attempting to parsing it
        # So, how to detect the date in a log line?, how to extract it?
        # and how to parse it later? are open questions
        left, right = date_string.split('.')
        idx = max(right.find('-'), right.find('+'))
        date_string = left + right[idx:].replace(":", "")
        date = datetime.datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S%z')
        date = date.replace(microsecond=int(right[:idx]))

        # TODO: it is possible in theory split this into more "streams"
        # and apply a different compression for each one but the few
        # experiments that I ran shows that the compression of the 'text'
        # part is still dominating the size of the compressed files.
        # TODO: How the decoder will know how to get back the original
        # str-like text? Perhaps can be arbitrary chosen by the decoder.
        # TODO: we assume that the text ends in a newline and this is critical
        # as we can use it for
        # line delimiter during the decompression. It would be much
        # better using an 'universal' separator that we can know that
        # it is not present in the original text after it was encoded
        # as bytes
        yield date, text.encode(encoding)

def join(tuples, encoding='utf8'):
    ''' Join the fields of <tuples> representing the <dates> and <texts>
        to reconstruct the log lines.

        This is roughly the inverse of zlog.split.

        >>> tuples = zip(dates, (t[:-1] for t in texts))
        >>> lines = zlog.join(tuples)
        >>> print(''.join(lines))
        2019-07-13T01:17:03.034216+00:00 host text e637-39f180263185
        2019-07-13T01:17:03.041246+00:00 host text 0263185
        2019-07-13T01:17:06.041452+00:00 host text 3185
        2019-07-13T01:19:03.042579+00:00 host text (queue size 0)

    '''
    for date, text in tuples:
        # TODO: like in zlog.split we are assuming that the date
        # is the first element followed by a single space and then by
        # the rest of the text ending in a new line.
        # Not all the log lines are like this.

        # TODO: the zlog.split looses the format of the date, microseconds and
        # the timezone.
        # We cannot reconstruct the log line how it was.
        yield "%s %s\n" % (date.isoformat(), text.decode(encoding))

def delta_encode(dates):
    ''' Encode the dates as the first-order differences and return
        an iterator of the seconds between the consecutive dates.

        >>> deltas = list(zlog.delta_encode(dates))
        >>> deltas
        [1562980623034216, 7030, 3000206, 117001127]
    '''

    # TODO: taking the "principle of the times" as starting point
    # seems too much, but I don't know if other value could have
    # any real impact in the performance.
    # And we must be sure that zlog.delta_decode uses the same
    # point of reference
    prev_date = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)

    dt = datetime.timedelta(microseconds=1)
    for date in dates:
        delta = date - prev_date

        # TODO: microseconds precision
        yield int(delta / dt)
        prev_date = date

def delta_decode(deltas):
    ''' Decode the original dates from their first-order differences.

        This is roughly the inverse of zlog.delta_encode

        >>> dates = zlog.delta_decode(deltas)
        >>> [d.isoformat() for d in dates]
        ['2019-07-13T01:17:03.034216+00:00',
         '2019-07-13T01:17:03.041246+00:00',
         '2019-07-13T01:17:06.041452+00:00',
         '2019-07-13T01:19:03.042579+00:00']
    '''
    accum = 0
    for delta in deltas:
        accum += delta

        secs = accum // 1000000
        frac = accum % 1000000

        date = datetime.datetime.fromtimestamp(secs)
        date = date.replace(microsecond=frac, tzinfo=datetime.timezone.utc)
        yield date

def delta_to_bytes(deltas):
    ''' Pack the deltas as bytes.

        >>> raw_deltas = list(zlog.delta_to_bytes(deltas))
        >>> raw_deltas
        ['\x00\x05\x8d\x85\xc8\xd3\xa7h',
         '\x00\x00\x00\x00\x00\x00\x1bv',
         '\x00\x00\x00\x00\x00-\xc7\x8e',
         '\x00\x00\x00\x00\x06\xf9K\xa7']
    '''
    for delta in deltas:
        # TODO: 8 bytes? we may improve this with techniques like
        # frame of reference and packaging more tight the bits
        yield struct.pack('>Q', delta)

def delta_from_bytes(bchunks):
    ''' Assemble the deltas from the chunk of bytes.
        The chunks *not* necessary are aligned.

        >>> raw_deltas[1:2] = [raw_deltas[1][:2], raw_deltas[1][2:]]
        >>> deltas = list(zlog.delta_from_bytes(raw_deltas))
        >>> deltas
        [1562980623034216, 7030, 3000206, 117001127]
    '''
    last_bytes = b''
    for bchunk in bchunks:
        last_bytes += bchunk
        n = len(last_bytes) // 8
        rem = len(last_bytes) % 8

        if n:
            deltas = struct.unpack('>%iQ%ix' % (n, rem), last_bytes)

            for delta in deltas:
                yield delta

        if rem:
            last_bytes = last_bytes[-rem:]
        else:
            last_bytes = b''

def compress_lzma(chunks, lvl):
    ''' Compress the given list of <chunks> using <lvl> as the level
        of the compression (1 for the fastest but with lowest compression
        ratio, 9 for the slowest but with the highest compression ratio;
        0 means 'no compression at all'.

        >>> chunks = list(zlog.compress_lzma(texts, lvl=9))
    '''
    c = lzma.LZMACompressor(preset=lvl)
    for chunk in chunks:
        data = c.compress(chunk)
        if data:
            yield data

    data = c.flush()
    if data:
        yield data

def decompress_lzma(compressed_data):
    ''' Decompress the given <compressed_data> returning an iterator
        of the decompressed texts.

        Keep in mind that the chunks returned have *arbitrary* lengths,
        in particular they will *not* end in a newline.

        >>> texts = list(zlog.decompress_lzma(chunks))
        >>> texts
        ['host text e637-39f180263185\nhost text 0263185\nhost text 3185\nhost text ('
         'queue size 0)\n']

        >>> isinstance(texts[0], bytes)
        True
    '''
    d = lzma.LZMADecompressor()
    for data in compressed_data:
        chunk = d.decompress(data)
        if chunk:
            yield chunk

def write_to_disk(chunks, fname, binary):
    ''' Open <fname> and write the given chunks.
        Open the file in binary or text mode (<binary>=False)

        This must be the end of a pipeline of iterators/generators
        '''
    flags = 'wb' if binary else 'wt'
    with open(fname, flags) as f:
        for chunk in chunks:
            f.write(chunk)

def read_from_disk(fname, binary, blk):
    ''' Open <fname> and read from it:
         - chunks of data of size up to <blk> if <binary> is True
         - lines of text if <binary> is False

        This must be the begin of a pipeline of iterators/generators
    '''
    flags = 'rb' if binary else 'rt'
    with open(fname, flags) as f:
        if binary:
            chunk = f.read(blk)
            while chunk:
                yield chunk
                chunk = f.read(blk)
        else:
            for line in f:
                yield line

def assemble_lines(chunks):
    ''' Assemble from the <chunks> of texts lines.

        >>> texts = list(zlog.assemble_lines(texts))
        >>> texts
        ['host text e637-39f180263185',
         'host text 0263185',
         'host text 3185',
         'host text (queue size 0)']
        '''
    last_partial_line = []
    for chunk in chunks:
        # TODO: we are assuming that the new line was the separator
        # between lines, I'm not sure if this is universal (see zlog.split)
        new_lines = chunk.split(b'\n')

        # if we have some chunks from the last incomplete
        # line, join the "first" line with them as it is
        # the continuation of that incomplete line
        if last_partial_line:
            last_partial_line.append(new_lines[0])
            del new_lines[0]

        if new_lines:
            # if we have new lines (even after removing the first one)
            # that means that the last incomplete line is complete
            # (if we have one of course), flush it
            if last_partial_line:
                yield (b''.join(last_partial_line))

            # if the last new line is not empty, that means that it is
            # and incomplete line, otherwise we are "line aligned".
            if new_lines[-1]:
                last_partial_line = [new_lines[-1]]
            else:
                last_partial_line = []
            del new_lines[-1]

        # flush all the remaining new lines
        for line in new_lines:
            yield line

    if last_partial_line:
        yield (b''.join(last_partial_line))

def compress_dates_pipeline(dates, lvl, fname):
    deltas = delta_encode(dates)
    data = delta_to_bytes(deltas)
    cdata = compress_lzma(data, lvl)
    write_to_disk(cdata, fname, binary=True)

def decompress_dates_pipeline(fname, blk=4*1024):
    cdata = read_from_disk(fname, binary=True, blk=blk)
    data = decompress_lzma(cdata)
    deltas = delta_from_bytes(data)
    dates = delta_decode(deltas)

    return dates

def compress_texts_pipeline(texts, lvl, fname):
    cdata = compress_lzma(texts, lvl)
    write_to_disk(cdata, fname, binary=True)

def decompress_texts_pipeline(fname, blk=4*1024):
    cdata = read_from_disk(fname, binary=True, blk=blk)
    data = decompress_lzma(cdata)
    lines = assemble_lines(data)

    return lines

def compress_log_file_pipeline(fname_dates, fname_texts, fname_input, lvl):
    lines = read_from_disk(fname_input, binary=False, blk=0)
    tuples = split(lines)

    cdates_pipeline = functools.partial(
                        compress_dates_pipeline, lvl=lvl, fname=fname_dates)

    ctexts_pipeline = functools.partial(
                        compress_texts_pipeline, lvl=lvl, fname=fname_texts)

    fan_out(tuples, cdates_pipeline, ctexts_pipeline, maxsize=1024)

def decompress_log_file_pipeline(fname_dates, fname_texts, fname_output):
    ddates_pipeline = functools.partial(
                            decompress_dates_pipeline, fname=fname_dates)
    dtexts_pipeline = functools.partial(
                            decompress_texts_pipeline, fname=fname_texts)

    tuples = fan_in(ddates_pipeline, dtexts_pipeline, maxsize=1024)

    lines = join(tuples)
    write_to_disk(lines, fname_output, binary=False)

def compress_log_not_split(fname_output, fname_input, lvl):
    lines = read_from_disk(fname_input, binary=False, blk=0)
    compress_texts_pipeline((l.encode('utf8') for l in lines), lvl, fname_output)

def decompress_log_not_split(fname_output, fname_input):
    lines = decompress_texts_pipeline(fname_input)
    write_to_disk((l.decode('utf8')+'\n' for l in lines), fname_output, binary=False)

def fan_out(tuples, *funcs, maxsize=512):
    ''' Iterate over <tuples>, and for each tuple read, call
        f0(tuple[0]), f1(tuple[1]), and so on: the number
        of fields in each tuple *must* be each to the number of <funcs>.

        The processing are run in parallel using threads and the arguments
        (tuple[i]) are sent using queues.

        This function blocks until <tuples> iterator is exhausted,
        the items are processed by <funcs> and the threads are done (joined).

        Outputs from the <funcs> are *not* collected, this needs to be
        done by the caller.

        >>> items = ((i, i+1)  for i in range(0, 32, 2))
        >>> even_results = []
        >>> odd_results  = []

        >>> def even(it):
        ...     for i in it:
        ...         even_results.append(i)

        >>> def odd(it):
        ...     for i in it:
        ...         odd_results.append(i)

        >>> zlog.fan_out(items, even, odd, maxsize=5)
        >>> even_results
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]

        >>> odd_results
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        '''
    def queue_as_iterator(_q):
        i = _q.get()
        while i is not None:
            yield i
            _q.task_done()
            i = _q.get()

        _q.task_done()

    def worker(_f, _q):
        _f(queue_as_iterator(_q))

    n = len(funcs)
    queues = [queue.Queue(maxsize) for _ in range(n)]

    # spawn threads
    threads = []
    for f, q in zip(funcs, queues):
        th = threading.Thread(target=worker, args=(f, q))
        th.start()
        threads.append(th)

    # send that data in a round robin fashion, consume the <tuples> iterator
    for t in tuples:
        for i, d in enumerate(t):
            queues[i].put(d)

    # send a last round with Nones to mark the end of the processing
    for i in range(n):
        queues[i].put(None)

    # join and clean up
    for th in threads:
        th.join()

    for q in queues:
        q.join()

def fan_in(*funcs, maxsize=1024):
    ''' Run each function in <funcs> in parallel collecting and
        yielding their results as tuples of the form (r0, r1, ...)
        where r0 is the result of f0, r1 of f1 and so on.

        >>> def even():
        ...     for i in range(0, 32, 2):
        ...         yield i

        >>> def odd():
        ...     for i in range(0, 32, 2):
        ...         yield i+1

        >>> list(zlog.fan_in(even, odd, maxsize=5))
        [(0, 1),
         (2, 3),
         (4, 5),
         (6, 7),
         (8, 9),
         (10, 11),
         (12, 13),
         (14, 15),
         (16, 17),
         (18, 19),
         (20, 21),
         (22, 23),
         (24, 25),
         (26, 27),
         (28, 29),
         (30, 31)]
    '''
    def worker(_f, _q):
        for _r in _f():
            _q.put(_r)

        _q.put(None)

    n = len(funcs)
    queues = [queue.Queue(maxsize) for _ in range(n)]

    # spawn threads
    threads = []
    for f, q in zip(funcs, queues):
        th = threading.Thread(target=worker, args=(f, q))
        th.start()
        threads.append(th)

    # get the results
    round = tuple(q.get() for q in queues)
    while round[0] is not None:
        yield round
        [q.task_done() for q in queues]
        round = tuple(q.get() for q in queues)

    [q.task_done() for q in queues]

    # join and clean up
    for th in threads:
        th.join()

    for q in queues:
        q.join()

def usage():
    print("""Usage:
  zlog c <lvl> <fname>
  Compress the file <fname> at <lvl> level and generates the compressed files
  with names <fname>.<x>.zlog where <x> is <d> (dates) and <t> (texts)

  zlog d <fname> <fname-out>
  Decompress the files <fname>.<x>.zlog and save the decompressed file
  in <fname-out>.

  Use C and D in replace of c and d to compress / decompress the logs
  using a standard LZMA compressor.
""")
    print(sys.argv)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
        sys.exit(1)

    op = sys.argv[1]
    if op == 'c':
        lvl = int(sys.argv[2])
        fname = sys.argv[3]

        fname_dates = fname + '.d.zlog'
        fname_texts = fname + '.t.zlog'

        compress_log_file_pipeline(fname_dates, fname_texts, fname, lvl)

    elif op == 'd':
        fname = sys.argv[2]
        fname_output = sys.argv[3]

        fname_dates = fname + '.d.zlog'
        fname_texts = fname + '.t.zlog'

        decompress_log_file_pipeline(fname_dates, fname_texts, fname_output)

    elif op == 'C':
        lvl = int(sys.argv[2])
        fname = sys.argv[3]

        fname_output = fname + '.0.zlog'

        compress_log_not_split(fname_output, fname, lvl)

    elif op == 'D':
        fname = sys.argv[2]
        fname_output = sys.argv[3]

        fname_input = fname + '.0.zlog'

        decompress_log_not_split(fname_output, fname_input)

    else:
        usage()
        sys.exit(1)

