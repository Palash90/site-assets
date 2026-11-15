function getLatestDate(date1, date2) {
    // Check if both dates are null or undefined
    if (!date1 && !date2) {
        return null;
    }

    // If only one date is null or undefined, return the other date
    if (!date1) {
        return date2;
    }

    if (!date2) {
        return date1;
    }

    // Compare the dates and return the latest one
    return date1 > date2 ? date1 : date2;
}

// Date-only publishDate defaults to 16:00 IST (10:30 UTC) so scheduled posts
// release at a consistent afternoon hour regardless of build machine timezone.
// A publishDate that already carries a time (e.g. "2026-11-18T16:00:00+05:30")
// is respected as-is.
const DEFAULT_RELEASE_UTC_HOUR = 10; // 16:00 IST
const DEFAULT_RELEASE_UTC_MINUTE = 30;

const parsePublishDate = (value) => {
    if (!value) {
        return undefined;
    }

    const str = String(value);

    // Author specified a time - respect it exactly as written.
    if (/\d{1,2}:\d{2}/.test(str)) {
        return new Date(str);
    }

    let year, month, day;
    if (/^\d{4}-\d{2}-\d{2}$/.test(str)) {
        [year, month, day] = str.split('-').map(Number);
    } else {
        const d = new Date(str);
        if (isNaN(d.getTime())) {
            return undefined;
        }
        // Date-only strings parse as local midnight; recover the authored calendar date.
        year = d.getFullYear();
        month = d.getMonth() + 1;
        day = d.getDate();
    }

    return new Date(Date.UTC(year, month - 1, day, DEFAULT_RELEASE_UTC_HOUR, DEFAULT_RELEASE_UTC_MINUTE));
}

const getDateString = (date) => {
    if (!date) {
        return date
    }

    let year = new Intl.DateTimeFormat('en', { year: 'numeric' }).format(date);
    let month = new Intl.DateTimeFormat('en', { month: 'short' }).format(date);
    let day = new Intl.DateTimeFormat('en', { day: '2-digit' }).format(date);

    return `${month} ${day}, ${year}`;
}

const modifyArray = (arr) => {

    arr = arr.map(c => {
        c = { ...c }
        c.publishDate = parsePublishDate(c.publishDate)
        c.lastUpdated = c.lastUpdated ? new Date(c.lastUpdated) : undefined

        c.sortBy = getLatestDate(c.publishDate, c.lastUpdated)
        return c
    })

    arr.sort((a, b) => b.sortBy - a.sortBy)

    return arr.map(c => {
        return {
            ...c,
            publishDate: getDateString(c.publishDate),
            lastUpdated: getDateString(c.lastUpdated),
        }
    })
}

