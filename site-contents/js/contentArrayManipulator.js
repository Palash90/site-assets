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

const getDateString = (date) => {
    if (!date) {
        return date
    }

    let year = new Intl.DateTimeFormat('en', { year: 'numeric' }).format(date);
    let month = new Intl.DateTimeFormat('en', { month: 'short' }).format(date);
    let day = new Intl.DateTimeFormat('en', { day: '2-digit' }).format(date);

    return `${day}-${month}-${year}`;
}

const modifyArray = (arr) => {

    arr = arr.map(c => {
        c.publishDate = c.publishDate ? new Date(c.publishDate) : undefined
        c.lastUpdated = c.lastUpdated ? new Date(c.lastUpdated) : undefined

        c.sortBy = getLatestDate(c.publishDate, c.lastUpdated)
        return c
    })

    arr.sort((a, b) => b.sortBy - a.sortBy)

    return arr.map(c => {
        return {
            id: c.id,
            title: c.title,
            publishDate: getDateString(c.publishDate),
            lastUpdated: getDateString(c.lastUpdated),
            mdUrl: c.mdUrl,
            videoId: c.videoId
        }
    })
}

