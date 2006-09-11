# Format a datetime through its full proleptic Gregorian date range.
#
# >>> strftime(datetime.date(1850, 8, 2), "%Y/%M/%d was a %A")
# '1850/00/02 was a Friday'
# >>>

import re, time

# remove the unsupposed "%s" command.  But don't
# do it if there's an even number of %s before the s
# because those are all escaped.  Can't simply
# remove the s because the result of
#  %sY
# should be %Y if %s isn't supported, not the
# 4 digit year.
_illegal_s = re.compile(r"((^|[^%])(%%)*%s)")

def _findall(text, substr):
     # Also finds overlaps
     sites = []
     i = 0
     while 1:
         j = text.find(substr, i)
         if j == -1:
             break
         sites.append(j)
         i=j+1
     return sites

# Every 28 years the calendar repeats, except through century leap
# years where it's 6 years.  But only if you're using the Gregorian
# calendar.  ;)

def strftime(dt, fmt):
    if _illegal_s.search(fmt):
        raise TypeError("This strftime implementation does not handle %s")
    # don't use strftime method at all.
    #if dt.year > 1900:
    #    return dt.strftime(fmt)

    year = dt.year
    # For every non-leap year century, advance by
    # 6 years to get into the 28-year repeat cycle
    delta = 2000 - year
    off = 6*(delta // 100 + delta // 400)
    year = year + off

    # Move to around the year 2000
    year = year + ((2000 - year)//28)*28
    timetuple = dt.timetuple()
    s1 = time.strftime(fmt, (year,) + timetuple[1:])
    sites1 = _findall(s1, str(year))
    
    s2 = time.strftime(fmt, (year+28,) + timetuple[1:])
    sites2 = _findall(s2, str(year+28))

    sites = []
    for site in sites1:
        if site in sites2:
            sites.append(site)
            
    s = s1
    syear = "%4d" % (dt.year,)
    for site in sites:
        s = s[:site] + syear + s[site+4:]
    return s
