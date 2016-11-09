//setup API
var DisneyAPI = require("wdwjs");

var MagicKingdom = new DisneyAPI.WaltDisneyWorldMagicKingdom();

//Get Magic Kingdom wait times
MagicKingdom.GetWaitTimes(function(err, data){
	if (err) return console.error("Error fetching Magic Kingdom wait times: " +err);
	console.log(JSON.stringify(data, null, 2));
    });

//