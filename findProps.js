function findProp(prop) {
    var obj = data
    prop = prop.split('.');
    for (var i = 0; i < prop.length; i++) {
        if (typeof obj[prop[i]] == 'undefined')
            return null;
        obj = obj[prop[i]];
    }
    return obj;
}