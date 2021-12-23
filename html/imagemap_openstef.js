// SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com>
// SPDX-License-Identifier: MPL-2.0

var map = $('#OPENSTEF')
var captions = {
        total_load: ["Total Load",
                      "Total load for a specific substation which can be selected above. "
                      + "Last part of the time series in the forecast (part with green area "
                      + "under the curve)"],
		forecast_realised: ["Forecast and Realised",
                      "Realised load and forecasted load of different lead times. "
                      + "Last forecast is also shown. "
                      + "Values are given with an certain confidence band"],
		predictor_split: ["Predictors",
                      "Predictor for a specific lead time in action. "
                      + "Splitting of data (train, test and validation,  "
                      + "used to train the model with the specific horizon. "
		      + "Splitted data is shown for the realised and the forecasted data. "],
		information: ["Information",
                      "Information regarding the substation and forecast. "
                      + "Information containing the last measurement, days available of prediction, "
		      + "last update of the information, the state of the forecast quality ( "
                      + "actual, not renewed or substituted), EAN for the substation (if available) "
		      + "and the API for the substation (if available)"],
		parameters: ["Adjustable input for the dashboard",
                      "Values that can be selected in order to view the desired input: "
                      + "\n * Prediction: location of substation "
                      + "\n * PID: unique integer value related to the substation "
                      + "\n * Confidence: used confidence interval for forecasting "
                      + "\n * SpecificTAhead: select forecast made at specified lead time"
                      + "\n * PID: unique integer value related to the substation "],
		location:["Location related to the substation",
                      "Location is specified by latitude and longitude. "],
		feature_importance:["Feature Importance",
                      "Importance of features used in the trained model for forecasting. "
                      + "The larger the surface related to the feature, "
                      + "the more important the feature is for the forecasting of "
                      + "the specific substation, importance and weight of feature "
                      + "is given in percentage. "],
		suspicious_input: ["Suspicious inputs",
                      "Periods with suspicious input data is detected, "
                      + "which are highlighted with a red colored area."],
		energy_split_plot: ["Split components",
                      "Forecast is split into solar (pv) , wind and other energy. "
                      + "Plotted for those categories, including the main forecast"],
		energy_split_tab: ["Energy splitting coefficients",
                      "In order to split forecast into solar, wind and other energy, "
                      + "coefficients must be specified. In the table the coefficients "
                      + "are given for each component (wind solar and energy profiles). "
                      + "The time when coefficients were calculate, is also shown"],
            }
var single_opts = {
                fillColor: '000000',
                fillOpacity: 0.1,
                stroke: true,
                strokeColor: 'ff0100',
                strokeWidth: 1
            }
var all_opts = {
                fillColor: 'ffffff',
                fillOpacity: 0.1,
                stroke: true,
                strokeWidth: 2,
                strokeColor: 'ffffff'
            }
var initial_opts = {
                mapKey: 'data-name',
                isSelectable: false,
                onMouseover: function (data) {
                    var inArea = true;
                    $('#OPENSTEF-caption-header').text(captions[data.key][0]);
                    $('#OPENSTEF-caption-text').text(captions[data.key][1]);
                    $('#OPENSTEF-caption').show();
                },
                onMouseout: function (data) {
                    var inArea = false;
                    $('#OPENSTEF-caption').hide();
                }
            };
var opts = $.extend({}, all_opts, initial_opts, single_opts);

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
