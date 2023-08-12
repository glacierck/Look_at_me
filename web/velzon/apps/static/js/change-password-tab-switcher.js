/* global $, console */
$(document).ready(function () {
    "use strict";
    var hash = window.location.hash;
    console.log('Hash value:', hash);
    if (hash) {
        var targetLink = $('a[href="' + hash + '"]');
        console.log('Target link:', targetLink);
        targetLink.click();
    }
});

