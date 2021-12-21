// javascript
var inArea,
            map = $('#OPENSTEF'),
            captions = {
                total_load: ["Total Load",
                      "Total load for a specific substation which can be set above. "
                      + "Last part of the time series in the prognose (part with green area "
                      + "under the curve)"],
		forecast_realised: ["Forecast and Realised",
                      "Realised load and forecasted load of different timestamps ahead. "
                      + "Last forecast is also shown. "
                      + "Values are given with an certain confidence band"],
		predictor_split: ["Predictors",
                      "Predictor for a specific horizon in action. "
                      + "Splitting of data (train, test and validation,  "
                      + "used to train the model with the specific horizon. "
		      + "Splitted data is specified for the realised and the forecasted data. "],
		information: ["Information",
                      "Information regarding the substation and forecast. "
                      + "Information contain the last measurement, days available of prediction, "
		      + "last update of the information, the state of the forecast quality ( "
                      + "actual or not renewed), EAN for the substation (if available) and "
		      + "the API for the substation (if available)"],
		parameters: ["Adjustable input for the dashboard",
                      "Values that can be adjusted in order to select the wanted input: "
                      + "\n * Prediction: location of substation "
                      + "\n * PID: unique integer value related to the substation "
                      + "\n * Confidence: used confidence interval for forecasting "
                      + "\n * SpecificTAhead: select the wanted time stamps ahead as start point"
                      + "\n * PID: unique integer value related to the substation "],
		location:["Location related to the substation",
                      "Location is specified by latitude and longitude. "],
		feature_importance:["Feature Importance",
                      "Importance of features used in the trained model for forecasting. "
                      + "The larger the surface related to the feature, "
                      + "the more important the feature is for the forecasting of "
                      + "the specific substation, importance and weight of feature "
                      + "is given in percentage. "],
		flatliners: ["Presence of flatliners",
                      "If flatliners are present in the data, "
                      + "they are highlighted with an red colored area for the given period."],
		energy_split_plot: ["Split components",
                      "Forecast is split into solar (pv) , wind and other energy. "
                      + "Plotted for those categories, including the main forecast"],
		energy_split_tab: ["Energy splitting coefficients",
                      "In order to split forecast into solar, wind and other energy, "
                      + "coefficients must be specified. In the table the coefficients "
                      + "are fiven for each component (wind solar and energy profiles). "
                      + "The time when coefficients were calculate, is specified"],
            },
            single_opts = {
                fillColor: '000000',
                fillOpacity: 0.1,
                stroke: true,
                strokeColor: 'ff0100',
                strokeWidth: 1
            },
            all_opts = {
                fillColor: 'ffffff',
                fillOpacity: 0.1,
                stroke: true,
                strokeWidth: 2,
                strokeColor: 'ffffff'
            },
            initial_opts = {
                mapKey: 'data-name',
                isSelectable: false,
                onMouseover: function (data) {
                    inArea = true;
                    $('#OPENSTEF-caption-header').text(captions[data.key][0]);
                    $('#OPENSTEF-caption-text').text(captions[data.key][1]);
                    $('#OPENSTEF-caption').show();
                },
                onMouseout: function (data) {
                    inArea = false;
                    $('#OPENSTEF-caption').hide();
                }
            };
        opts = $.extend({}, all_opts, initial_opts, single_opts);

        map.mapster('unbind')
            .mapster(opts)
            .bind('mouseover', function () {
                if (!inArea) {
                    map.mapster('set_options', all_opts)
                       .mapster('set', true, 'all')
                       .mapster('set_options', single_opts);
                }
            }).bind('mouseout', function () {
                if (!inArea) {
                    map.mapster('set', false, 'all');
                }
            });