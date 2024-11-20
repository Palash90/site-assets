function hasDuplicateProperty(arr, prop) {
    const seen = new Set();
    for (const item of arr) {
        if (seen.has(item[prop])) {
            console.log(item[prop], "has duplicate entry")
            return true; // Duplicate found
        }
        seen.add(item[prop]);
    }
    return false; // No duplicates
}

var allContents = sweContents.concat(musicContents).concat(drafts)

if(hasDuplicateProperty(allContents, "id")){
    throw Error("Duplicate blog id found");
}

const contents = {
    swe: sweContents,
    music: musicContents,
    drafts: drafts
}