<!DOCTYPE html>
<html>
<head><meta charset="utf-8" />

<title>Titanic</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>



<style type="text/css">
    /*!
*
* Twitter Bootstrap
*
*/
/*!
 * Bootstrap v3.3.7 (http://getbootstrap.com)
 * Copyright 2011-2016 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/master/LICENSE)
 */
/*! normalize.css v3.0.3 | MIT License | github.com/necolas/normalize.css */
html {
  font-family: sans-serif;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
}
body {
  margin: 0;
}
article,
aside,
details,
figcaption,
figure,
footer,
header,
hgroup,
main,
menu,
nav,
section,
summary {
  display: block;
}
audio,
canvas,
progress,
video {
  display: inline-block;
  vertical-align: baseline;
}
audio:not([controls]) {
  display: none;
  height: 0;
}
[hidden],
template {
  display: none;
}
a {
  background-color: transparent;
}
a:active,
a:hover {
  outline: 0;
}
abbr[title] {
  border-bottom: 1px dotted;
}
b,
strong {
  font-weight: bold;
}
dfn {
  font-style: italic;
}
h1 {
  font-size: 2em;
  margin: 0.67em 0;
}
mark {
  background: #ff0;
  color: #000;
}
small {
  font-size: 80%;
}
sub,
sup {
  font-size: 75%;
  line-height: 0;
  position: relative;
  vertical-align: baseline;
}
sup {
  top: -0.5em;
}
sub {
  bottom: -0.25em;
}
img {
  border: 0;
}
svg:not(:root) {
  overflow: hidden;
}
figure {
  margin: 1em 40px;
}
hr {
  box-sizing: content-box;
  height: 0;
}
pre {
  overflow: auto;
}
code,
kbd,
pre,
samp {
  font-family: monospace, monospace;
  font-size: 1em;
}
button,
input,
optgroup,
select,
textarea {
  color: inherit;
  font: inherit;
  margin: 0;
}
button {
  overflow: visible;
}
button,
select {
  text-transform: none;
}
button,
html input[type="button"],
input[type="reset"],
input[type="submit"] {
  -webkit-appearance: button;
  cursor: pointer;
}
button[disabled],
html input[disabled] {
  cursor: default;
}
button::-moz-focus-inner,
input::-moz-focus-inner {
  border: 0;
  padding: 0;
}
input {
  line-height: normal;
}
input[type="checkbox"],
input[type="radio"] {
  box-sizing: border-box;
  padding: 0;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: textfield;
  box-sizing: content-box;
}
input[type="search"]::-webkit-search-cancel-button,
input[type="search"]::-webkit-search-decoration {
  -webkit-appearance: none;
}
fieldset {
  border: 1px solid #c0c0c0;
  margin: 0 2px;
  padding: 0.35em 0.625em 0.75em;
}
legend {
  border: 0;
  padding: 0;
}
textarea {
  overflow: auto;
}
optgroup {
  font-weight: bold;
}
table {
  border-collapse: collapse;
  border-spacing: 0;
}
td,
th {
  padding: 0;
}
/*! Source: https://github.com/h5bp/html5-boilerplate/blob/master/src/css/main.css */
@media print {
  *,
  *:before,
  *:after {
    background: transparent !important;
    box-shadow: none !important;
    text-shadow: none !important;
  }
  a,
  a:visited {
    text-decoration: underline;
  }
  a[href]:after {
    content: " (" attr(href) ")";
  }
  abbr[title]:after {
    content: " (" attr(title) ")";
  }
  a[href^="#"]:after,
  a[href^="javascript:"]:after {
    content: "";
  }
  pre,
  blockquote {
    border: 1px solid #999;
    page-break-inside: avoid;
  }
  thead {
    display: table-header-group;
  }
  tr,
  img {
    page-break-inside: avoid;
  }
  img {
    max-width: 100% !important;
  }
  p,
  h2,
  h3 {
    orphans: 3;
    widows: 3;
  }
  h2,
  h3 {
    page-break-after: avoid;
  }
  .navbar {
    display: none;
  }
  .btn > .caret,
  .dropup > .btn > .caret {
    border-top-color: #000 !important;
  }
  .label {
    border: 1px solid #000;
  }
  .table {
    border-collapse: collapse !important;
  }
  .table td,
  .table th {
    background-color: #fff !important;
  }
  .table-bordered th,
  .table-bordered td {
    border: 1px solid #ddd !important;
  }
}
@font-face {
  font-family: 'Glyphicons Halflings';
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot');
  src: url('../components/bootstrap/fonts/glyphicons-halflings-regular.eot?#iefix') format('embedded-opentype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff2') format('woff2'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.woff') format('woff'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.ttf') format('truetype'), url('../components/bootstrap/fonts/glyphicons-halflings-regular.svg#glyphicons_halflingsregular') format('svg');
}
.glyphicon {
  position: relative;
  top: 1px;
  display: inline-block;
  font-family: 'Glyphicons Halflings';
  font-style: normal;
  font-weight: normal;
  line-height: 1;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.glyphicon-asterisk:before {
  content: "\002a";
}
.glyphicon-plus:before {
  content: "\002b";
}
.glyphicon-euro:before,
.glyphicon-eur:before {
  content: "\20ac";
}
.glyphicon-minus:before {
  content: "\2212";
}
.glyphicon-cloud:before {
  content: "\2601";
}
.glyphicon-envelope:before {
  content: "\2709";
}
.glyphicon-pencil:before {
  content: "\270f";
}
.glyphicon-glass:before {
  content: "\e001";
}
.glyphicon-music:before {
  content: "\e002";
}
.glyphicon-search:before {
  content: "\e003";
}
.glyphicon-heart:before {
  content: "\e005";
}
.glyphicon-star:before {
  content: "\e006";
}
.glyphicon-star-empty:before {
  content: "\e007";
}
.glyphicon-user:before {
  content: "\e008";
}
.glyphicon-film:before {
  content: "\e009";
}
.glyphicon-th-large:before {
  content: "\e010";
}
.glyphicon-th:before {
  content: "\e011";
}
.glyphicon-th-list:before {
  content: "\e012";
}
.glyphicon-ok:before {
  content: "\e013";
}
.glyphicon-remove:before {
  content: "\e014";
}
.glyphicon-zoom-in:before {
  content: "\e015";
}
.glyphicon-zoom-out:before {
  content: "\e016";
}
.glyphicon-off:before {
  content: "\e017";
}
.glyphicon-signal:before {
  content: "\e018";
}
.glyphicon-cog:before {
  content: "\e019";
}
.glyphicon-trash:before {
  content: "\e020";
}
.glyphicon-home:before {
  content: "\e021";
}
.glyphicon-file:before {
  content: "\e022";
}
.glyphicon-time:before {
  content: "\e023";
}
.glyphicon-road:before {
  content: "\e024";
}
.glyphicon-download-alt:before {
  content: "\e025";
}
.glyphicon-download:before {
  content: "\e026";
}
.glyphicon-upload:before {
  content: "\e027";
}
.glyphicon-inbox:before {
  content: "\e028";
}
.glyphicon-play-circle:before {
  content: "\e029";
}
.glyphicon-repeat:before {
  content: "\e030";
}
.glyphicon-refresh:before {
  content: "\e031";
}
.glyphicon-list-alt:before {
  content: "\e032";
}
.glyphicon-lock:before {
  content: "\e033";
}
.glyphicon-flag:before {
  content: "\e034";
}
.glyphicon-headphones:before {
  content: "\e035";
}
.glyphicon-volume-off:before {
  content: "\e036";
}
.glyphicon-volume-down:before {
  content: "\e037";
}
.glyphicon-volume-up:before {
  content: "\e038";
}
.glyphicon-qrcode:before {
  content: "\e039";
}
.glyphicon-barcode:before {
  content: "\e040";
}
.glyphicon-tag:before {
  content: "\e041";
}
.glyphicon-tags:before {
  content: "\e042";
}
.glyphicon-book:before {
  content: "\e043";
}
.glyphicon-bookmark:before {
  content: "\e044";
}
.glyphicon-print:before {
  content: "\e045";
}
.glyphicon-camera:before {
  content: "\e046";
}
.glyphicon-font:before {
  content: "\e047";
}
.glyphicon-bold:before {
  content: "\e048";
}
.glyphicon-italic:before {
  content: "\e049";
}
.glyphicon-text-height:before {
  content: "\e050";
}
.glyphicon-text-width:before {
  content: "\e051";
}
.glyphicon-align-left:before {
  content: "\e052";
}
.glyphicon-align-center:before {
  content: "\e053";
}
.glyphicon-align-right:before {
  content: "\e054";
}
.glyphicon-align-justify:before {
  content: "\e055";
}
.glyphicon-list:before {
  content: "\e056";
}
.glyphicon-indent-left:before {
  content: "\e057";
}
.glyphicon-indent-right:before {
  content: "\e058";
}
.glyphicon-facetime-video:before {
  content: "\e059";
}
.glyphicon-picture:before {
  content: "\e060";
}
.glyphicon-map-marker:before {
  content: "\e062";
}
.glyphicon-adjust:before {
  content: "\e063";
}
.glyphicon-tint:before {
  content: "\e064";
}
.glyphicon-edit:before {
  content: "\e065";
}
.glyphicon-share:before {
  content: "\e066";
}
.glyphicon-check:before {
  content: "\e067";
}
.glyphicon-move:before {
  content: "\e068";
}
.glyphicon-step-backward:before {
  content: "\e069";
}
.glyphicon-fast-backward:before {
  content: "\e070";
}
.glyphicon-backward:before {
  content: "\e071";
}
.glyphicon-play:before {
  content: "\e072";
}
.glyphicon-pause:before {
  content: "\e073";
}
.glyphicon-stop:before {
  content: "\e074";
}
.glyphicon-forward:before {
  content: "\e075";
}
.glyphicon-fast-forward:before {
  content: "\e076";
}
.glyphicon-step-forward:before {
  content: "\e077";
}
.glyphicon-eject:before {
  content: "\e078";
}
.glyphicon-chevron-left:before {
  content: "\e079";
}
.glyphicon-chevron-right:before {
  content: "\e080";
}
.glyphicon-plus-sign:before {
  content: "\e081";
}
.glyphicon-minus-sign:before {
  content: "\e082";
}
.glyphicon-remove-sign:before {
  content: "\e083";
}
.glyphicon-ok-sign:before {
  content: "\e084";
}
.glyphicon-question-sign:before {
  content: "\e085";
}
.glyphicon-info-sign:before {
  content: "\e086";
}
.glyphicon-screenshot:before {
  content: "\e087";
}
.glyphicon-remove-circle:before {
  content: "\e088";
}
.glyphicon-ok-circle:before {
  content: "\e089";
}
.glyphicon-ban-circle:before {
  content: "\e090";
}
.glyphicon-arrow-left:before {
  content: "\e091";
}
.glyphicon-arrow-right:before {
  content: "\e092";
}
.glyphicon-arrow-up:before {
  content: "\e093";
}
.glyphicon-arrow-down:before {
  content: "\e094";
}
.glyphicon-share-alt:before {
  content: "\e095";
}
.glyphicon-resize-full:before {
  content: "\e096";
}
.glyphicon-resize-small:before {
  content: "\e097";
}
.glyphicon-exclamation-sign:before {
  content: "\e101";
}
.glyphicon-gift:before {
  content: "\e102";
}
.glyphicon-leaf:before {
  content: "\e103";
}
.glyphicon-fire:before {
  content: "\e104";
}
.glyphicon-eye-open:before {
  content: "\e105";
}
.glyphicon-eye-close:before {
  content: "\e106";
}
.glyphicon-warning-sign:before {
  content: "\e107";
}
.glyphicon-plane:before {
  content: "\e108";
}
.glyphicon-calendar:before {
  content: "\e109";
}
.glyphicon-random:before {
  content: "\e110";
}
.glyphicon-comment:before {
  content: "\e111";
}
.glyphicon-magnet:before {
  content: "\e112";
}
.glyphicon-chevron-up:before {
  content: "\e113";
}
.glyphicon-chevron-down:before {
  content: "\e114";
}
.glyphicon-retweet:before {
  content: "\e115";
}
.glyphicon-shopping-cart:before {
  content: "\e116";
}
.glyphicon-folder-close:before {
  content: "\e117";
}
.glyphicon-folder-open:before {
  content: "\e118";
}
.glyphicon-resize-vertical:before {
  content: "\e119";
}
.glyphicon-resize-horizontal:before {
  content: "\e120";
}
.glyphicon-hdd:before {
  content: "\e121";
}
.glyphicon-bullhorn:before {
  content: "\e122";
}
.glyphicon-bell:before {
  content: "\e123";
}
.glyphicon-certificate:before {
  content: "\e124";
}
.glyphicon-thumbs-up:before {
  content: "\e125";
}
.glyphicon-thumbs-down:before {
  content: "\e126";
}
.glyphicon-hand-right:before {
  content: "\e127";
}
.glyphicon-hand-left:before {
  content: "\e128";
}
.glyphicon-hand-up:before {
  content: "\e129";
}
.glyphicon-hand-down:before {
  content: "\e130";
}
.glyphicon-circle-arrow-right:before {
  content: "\e131";
}
.glyphicon-circle-arrow-left:before {
  content: "\e132";
}
.glyphicon-circle-arrow-up:before {
  content: "\e133";
}
.glyphicon-circle-arrow-down:before {
  content: "\e134";
}
.glyphicon-globe:before {
  content: "\e135";
}
.glyphicon-wrench:before {
  content: "\e136";
}
.glyphicon-tasks:before {
  content: "\e137";
}
.glyphicon-filter:before {
  content: "\e138";
}
.glyphicon-briefcase:before {
  content: "\e139";
}
.glyphicon-fullscreen:before {
  content: "\e140";
}
.glyphicon-dashboard:before {
  content: "\e141";
}
.glyphicon-paperclip:before {
  content: "\e142";
}
.glyphicon-heart-empty:before {
  content: "\e143";
}
.glyphicon-link:before {
  content: "\e144";
}
.glyphicon-phone:before {
  content: "\e145";
}
.glyphicon-pushpin:before {
  content: "\e146";
}
.glyphicon-usd:before {
  content: "\e148";
}
.glyphicon-gbp:before {
  content: "\e149";
}
.glyphicon-sort:before {
  content: "\e150";
}
.glyphicon-sort-by-alphabet:before {
  content: "\e151";
}
.glyphicon-sort-by-alphabet-alt:before {
  content: "\e152";
}
.glyphicon-sort-by-order:before {
  content: "\e153";
}
.glyphicon-sort-by-order-alt:before {
  content: "\e154";
}
.glyphicon-sort-by-attributes:before {
  content: "\e155";
}
.glyphicon-sort-by-attributes-alt:before {
  content: "\e156";
}
.glyphicon-unchecked:before {
  content: "\e157";
}
.glyphicon-expand:before {
  content: "\e158";
}
.glyphicon-collapse-down:before {
  content: "\e159";
}
.glyphicon-collapse-up:before {
  content: "\e160";
}
.glyphicon-log-in:before {
  content: "\e161";
}
.glyphicon-flash:before {
  content: "\e162";
}
.glyphicon-log-out:before {
  content: "\e163";
}
.glyphicon-new-window:before {
  content: "\e164";
}
.glyphicon-record:before {
  content: "\e165";
}
.glyphicon-save:before {
  content: "\e166";
}
.glyphicon-open:before {
  content: "\e167";
}
.glyphicon-saved:before {
  content: "\e168";
}
.glyphicon-import:before {
  content: "\e169";
}
.glyphicon-export:before {
  content: "\e170";
}
.glyphicon-send:before {
  content: "\e171";
}
.glyphicon-floppy-disk:before {
  content: "\e172";
}
.glyphicon-floppy-saved:before {
  content: "\e173";
}
.glyphicon-floppy-remove:before {
  content: "\e174";
}
.glyphicon-floppy-save:before {
  content: "\e175";
}
.glyphicon-floppy-open:before {
  content: "\e176";
}
.glyphicon-credit-card:before {
  content: "\e177";
}
.glyphicon-transfer:before {
  content: "\e178";
}
.glyphicon-cutlery:before {
  content: "\e179";
}
.glyphicon-header:before {
  content: "\e180";
}
.glyphicon-compressed:before {
  content: "\e181";
}
.glyphicon-earphone:before {
  content: "\e182";
}
.glyphicon-phone-alt:before {
  content: "\e183";
}
.glyphicon-tower:before {
  content: "\e184";
}
.glyphicon-stats:before {
  content: "\e185";
}
.glyphicon-sd-video:before {
  content: "\e186";
}
.glyphicon-hd-video:before {
  content: "\e187";
}
.glyphicon-subtitles:before {
  content: "\e188";
}
.glyphicon-sound-stereo:before {
  content: "\e189";
}
.glyphicon-sound-dolby:before {
  content: "\e190";
}
.glyphicon-sound-5-1:before {
  content: "\e191";
}
.glyphicon-sound-6-1:before {
  content: "\e192";
}
.glyphicon-sound-7-1:before {
  content: "\e193";
}
.glyphicon-copyright-mark:before {
  content: "\e194";
}
.glyphicon-registration-mark:before {
  content: "\e195";
}
.glyphicon-cloud-download:before {
  content: "\e197";
}
.glyphicon-cloud-upload:before {
  content: "\e198";
}
.glyphicon-tree-conifer:before {
  content: "\e199";
}
.glyphicon-tree-deciduous:before {
  content: "\e200";
}
.glyphicon-cd:before {
  content: "\e201";
}
.glyphicon-save-file:before {
  content: "\e202";
}
.glyphicon-open-file:before {
  content: "\e203";
}
.glyphicon-level-up:before {
  content: "\e204";
}
.glyphicon-copy:before {
  content: "\e205";
}
.glyphicon-paste:before {
  content: "\e206";
}
.glyphicon-alert:before {
  content: "\e209";
}
.glyphicon-equalizer:before {
  content: "\e210";
}
.glyphicon-king:before {
  content: "\e211";
}
.glyphicon-queen:before {
  content: "\e212";
}
.glyphicon-pawn:before {
  content: "\e213";
}
.glyphicon-bishop:before {
  content: "\e214";
}
.glyphicon-knight:before {
  content: "\e215";
}
.glyphicon-baby-formula:before {
  content: "\e216";
}
.glyphicon-tent:before {
  content: "\26fa";
}
.glyphicon-blackboard:before {
  content: "\e218";
}
.glyphicon-bed:before {
  content: "\e219";
}
.glyphicon-apple:before {
  content: "\f8ff";
}
.glyphicon-erase:before {
  content: "\e221";
}
.glyphicon-hourglass:before {
  content: "\231b";
}
.glyphicon-lamp:before {
  content: "\e223";
}
.glyphicon-duplicate:before {
  content: "\e224";
}
.glyphicon-piggy-bank:before {
  content: "\e225";
}
.glyphicon-scissors:before {
  content: "\e226";
}
.glyphicon-bitcoin:before {
  content: "\e227";
}
.glyphicon-btc:before {
  content: "\e227";
}
.glyphicon-xbt:before {
  content: "\e227";
}
.glyphicon-yen:before {
  content: "\00a5";
}
.glyphicon-jpy:before {
  content: "\00a5";
}
.glyphicon-ruble:before {
  content: "\20bd";
}
.glyphicon-rub:before {
  content: "\20bd";
}
.glyphicon-scale:before {
  content: "\e230";
}
.glyphicon-ice-lolly:before {
  content: "\e231";
}
.glyphicon-ice-lolly-tasted:before {
  content: "\e232";
}
.glyphicon-education:before {
  content: "\e233";
}
.glyphicon-option-horizontal:before {
  content: "\e234";
}
.glyphicon-option-vertical:before {
  content: "\e235";
}
.glyphicon-menu-hamburger:before {
  content: "\e236";
}
.glyphicon-modal-window:before {
  content: "\e237";
}
.glyphicon-oil:before {
  content: "\e238";
}
.glyphicon-grain:before {
  content: "\e239";
}
.glyphicon-sunglasses:before {
  content: "\e240";
}
.glyphicon-text-size:before {
  content: "\e241";
}
.glyphicon-text-color:before {
  content: "\e242";
}
.glyphicon-text-background:before {
  content: "\e243";
}
.glyphicon-object-align-top:before {
  content: "\e244";
}
.glyphicon-object-align-bottom:before {
  content: "\e245";
}
.glyphicon-object-align-horizontal:before {
  content: "\e246";
}
.glyphicon-object-align-left:before {
  content: "\e247";
}
.glyphicon-object-align-vertical:before {
  content: "\e248";
}
.glyphicon-object-align-right:before {
  content: "\e249";
}
.glyphicon-triangle-right:before {
  content: "\e250";
}
.glyphicon-triangle-left:before {
  content: "\e251";
}
.glyphicon-triangle-bottom:before {
  content: "\e252";
}
.glyphicon-triangle-top:before {
  content: "\e253";
}
.glyphicon-console:before {
  content: "\e254";
}
.glyphicon-superscript:before {
  content: "\e255";
}
.glyphicon-subscript:before {
  content: "\e256";
}
.glyphicon-menu-left:before {
  content: "\e257";
}
.glyphicon-menu-right:before {
  content: "\e258";
}
.glyphicon-menu-down:before {
  content: "\e259";
}
.glyphicon-menu-up:before {
  content: "\e260";
}
* {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
*:before,
*:after {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
html {
  font-size: 10px;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 13px;
  line-height: 1.42857143;
  color: #000;
  background-color: #fff;
}
input,
button,
select,
textarea {
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}
a {
  color: #337ab7;
  text-decoration: none;
}
a:hover,
a:focus {
  color: #23527c;
  text-decoration: underline;
}
a:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
figure {
  margin: 0;
}
img {
  vertical-align: middle;
}
.img-responsive,
.thumbnail > img,
.thumbnail a > img,
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  display: block;
  max-width: 100%;
  height: auto;
}
.img-rounded {
  border-radius: 3px;
}
.img-thumbnail {
  padding: 4px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: all 0.2s ease-in-out;
  -o-transition: all 0.2s ease-in-out;
  transition: all 0.2s ease-in-out;
  display: inline-block;
  max-width: 100%;
  height: auto;
}
.img-circle {
  border-radius: 50%;
}
hr {
  margin-top: 18px;
  margin-bottom: 18px;
  border: 0;
  border-top: 1px solid #eeeeee;
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
[role="button"] {
  cursor: pointer;
}
h1,
h2,
h3,
h4,
h5,
h6,
.h1,
.h2,
.h3,
.h4,
.h5,
.h6 {
  font-family: inherit;
  font-weight: 500;
  line-height: 1.1;
  color: inherit;
}
h1 small,
h2 small,
h3 small,
h4 small,
h5 small,
h6 small,
.h1 small,
.h2 small,
.h3 small,
.h4 small,
.h5 small,
.h6 small,
h1 .small,
h2 .small,
h3 .small,
h4 .small,
h5 .small,
h6 .small,
.h1 .small,
.h2 .small,
.h3 .small,
.h4 .small,
.h5 .small,
.h6 .small {
  font-weight: normal;
  line-height: 1;
  color: #777777;
}
h1,
.h1,
h2,
.h2,
h3,
.h3 {
  margin-top: 18px;
  margin-bottom: 9px;
}
h1 small,
.h1 small,
h2 small,
.h2 small,
h3 small,
.h3 small,
h1 .small,
.h1 .small,
h2 .small,
.h2 .small,
h3 .small,
.h3 .small {
  font-size: 65%;
}
h4,
.h4,
h5,
.h5,
h6,
.h6 {
  margin-top: 9px;
  margin-bottom: 9px;
}
h4 small,
.h4 small,
h5 small,
.h5 small,
h6 small,
.h6 small,
h4 .small,
.h4 .small,
h5 .small,
.h5 .small,
h6 .small,
.h6 .small {
  font-size: 75%;
}
h1,
.h1 {
  font-size: 33px;
}
h2,
.h2 {
  font-size: 27px;
}
h3,
.h3 {
  font-size: 23px;
}
h4,
.h4 {
  font-size: 17px;
}
h5,
.h5 {
  font-size: 13px;
}
h6,
.h6 {
  font-size: 12px;
}
p {
  margin: 0 0 9px;
}
.lead {
  margin-bottom: 18px;
  font-size: 14px;
  font-weight: 300;
  line-height: 1.4;
}
@media (min-width: 768px) {
  .lead {
    font-size: 19.5px;
  }
}
small,
.small {
  font-size: 92%;
}
mark,
.mark {
  background-color: #fcf8e3;
  padding: .2em;
}
.text-left {
  text-align: left;
}
.text-right {
  text-align: right;
}
.text-center {
  text-align: center;
}
.text-justify {
  text-align: justify;
}
.text-nowrap {
  white-space: nowrap;
}
.text-lowercase {
  text-transform: lowercase;
}
.text-uppercase {
  text-transform: uppercase;
}
.text-capitalize {
  text-transform: capitalize;
}
.text-muted {
  color: #777777;
}
.text-primary {
  color: #337ab7;
}
a.text-primary:hover,
a.text-primary:focus {
  color: #286090;
}
.text-success {
  color: #3c763d;
}
a.text-success:hover,
a.text-success:focus {
  color: #2b542c;
}
.text-info {
  color: #31708f;
}
a.text-info:hover,
a.text-info:focus {
  color: #245269;
}
.text-warning {
  color: #8a6d3b;
}
a.text-warning:hover,
a.text-warning:focus {
  color: #66512c;
}
.text-danger {
  color: #a94442;
}
a.text-danger:hover,
a.text-danger:focus {
  color: #843534;
}
.bg-primary {
  color: #fff;
  background-color: #337ab7;
}
a.bg-primary:hover,
a.bg-primary:focus {
  background-color: #286090;
}
.bg-success {
  background-color: #dff0d8;
}
a.bg-success:hover,
a.bg-success:focus {
  background-color: #c1e2b3;
}
.bg-info {
  background-color: #d9edf7;
}
a.bg-info:hover,
a.bg-info:focus {
  background-color: #afd9ee;
}
.bg-warning {
  background-color: #fcf8e3;
}
a.bg-warning:hover,
a.bg-warning:focus {
  background-color: #f7ecb5;
}
.bg-danger {
  background-color: #f2dede;
}
a.bg-danger:hover,
a.bg-danger:focus {
  background-color: #e4b9b9;
}
.page-header {
  padding-bottom: 8px;
  margin: 36px 0 18px;
  border-bottom: 1px solid #eeeeee;
}
ul,
ol {
  margin-top: 0;
  margin-bottom: 9px;
}
ul ul,
ol ul,
ul ol,
ol ol {
  margin-bottom: 0;
}
.list-unstyled {
  padding-left: 0;
  list-style: none;
}
.list-inline {
  padding-left: 0;
  list-style: none;
  margin-left: -5px;
}
.list-inline > li {
  display: inline-block;
  padding-left: 5px;
  padding-right: 5px;
}
dl {
  margin-top: 0;
  margin-bottom: 18px;
}
dt,
dd {
  line-height: 1.42857143;
}
dt {
  font-weight: bold;
}
dd {
  margin-left: 0;
}
@media (min-width: 541px) {
  .dl-horizontal dt {
    float: left;
    width: 160px;
    clear: left;
    text-align: right;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .dl-horizontal dd {
    margin-left: 180px;
  }
}
abbr[title],
abbr[data-original-title] {
  cursor: help;
  border-bottom: 1px dotted #777777;
}
.initialism {
  font-size: 90%;
  text-transform: uppercase;
}
blockquote {
  padding: 9px 18px;
  margin: 0 0 18px;
  font-size: inherit;
  border-left: 5px solid #eeeeee;
}
blockquote p:last-child,
blockquote ul:last-child,
blockquote ol:last-child {
  margin-bottom: 0;
}
blockquote footer,
blockquote small,
blockquote .small {
  display: block;
  font-size: 80%;
  line-height: 1.42857143;
  color: #777777;
}
blockquote footer:before,
blockquote small:before,
blockquote .small:before {
  content: '\2014 \00A0';
}
.blockquote-reverse,
blockquote.pull-right {
  padding-right: 15px;
  padding-left: 0;
  border-right: 5px solid #eeeeee;
  border-left: 0;
  text-align: right;
}
.blockquote-reverse footer:before,
blockquote.pull-right footer:before,
.blockquote-reverse small:before,
blockquote.pull-right small:before,
.blockquote-reverse .small:before,
blockquote.pull-right .small:before {
  content: '';
}
.blockquote-reverse footer:after,
blockquote.pull-right footer:after,
.blockquote-reverse small:after,
blockquote.pull-right small:after,
.blockquote-reverse .small:after,
blockquote.pull-right .small:after {
  content: '\00A0 \2014';
}
address {
  margin-bottom: 18px;
  font-style: normal;
  line-height: 1.42857143;
}
code,
kbd,
pre,
samp {
  font-family: monospace;
}
code {
  padding: 2px 4px;
  font-size: 90%;
  color: #c7254e;
  background-color: #f9f2f4;
  border-radius: 2px;
}
kbd {
  padding: 2px 4px;
  font-size: 90%;
  color: #888;
  background-color: transparent;
  border-radius: 1px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: bold;
  box-shadow: none;
}
pre {
  display: block;
  padding: 8.5px;
  margin: 0 0 9px;
  font-size: 12px;
  line-height: 1.42857143;
  word-break: break-all;
  word-wrap: break-word;
  color: #333333;
  background-color: #f5f5f5;
  border: 1px solid #ccc;
  border-radius: 2px;
}
pre code {
  padding: 0;
  font-size: inherit;
  color: inherit;
  white-space: pre-wrap;
  background-color: transparent;
  border-radius: 0;
}
.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}
.container {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
@media (min-width: 768px) {
  .container {
    width: 768px;
  }
}
@media (min-width: 992px) {
  .container {
    width: 940px;
  }
}
@media (min-width: 1200px) {
  .container {
    width: 1140px;
  }
}
.container-fluid {
  margin-right: auto;
  margin-left: auto;
  padding-left: 0px;
  padding-right: 0px;
}
.row {
  margin-left: 0px;
  margin-right: 0px;
}
.col-xs-1, .col-sm-1, .col-md-1, .col-lg-1, .col-xs-2, .col-sm-2, .col-md-2, .col-lg-2, .col-xs-3, .col-sm-3, .col-md-3, .col-lg-3, .col-xs-4, .col-sm-4, .col-md-4, .col-lg-4, .col-xs-5, .col-sm-5, .col-md-5, .col-lg-5, .col-xs-6, .col-sm-6, .col-md-6, .col-lg-6, .col-xs-7, .col-sm-7, .col-md-7, .col-lg-7, .col-xs-8, .col-sm-8, .col-md-8, .col-lg-8, .col-xs-9, .col-sm-9, .col-md-9, .col-lg-9, .col-xs-10, .col-sm-10, .col-md-10, .col-lg-10, .col-xs-11, .col-sm-11, .col-md-11, .col-lg-11, .col-xs-12, .col-sm-12, .col-md-12, .col-lg-12 {
  position: relative;
  min-height: 1px;
  padding-left: 0px;
  padding-right: 0px;
}
.col-xs-1, .col-xs-2, .col-xs-3, .col-xs-4, .col-xs-5, .col-xs-6, .col-xs-7, .col-xs-8, .col-xs-9, .col-xs-10, .col-xs-11, .col-xs-12 {
  float: left;
}
.col-xs-12 {
  width: 100%;
}
.col-xs-11 {
  width: 91.66666667%;
}
.col-xs-10 {
  width: 83.33333333%;
}
.col-xs-9 {
  width: 75%;
}
.col-xs-8 {
  width: 66.66666667%;
}
.col-xs-7 {
  width: 58.33333333%;
}
.col-xs-6 {
  width: 50%;
}
.col-xs-5 {
  width: 41.66666667%;
}
.col-xs-4 {
  width: 33.33333333%;
}
.col-xs-3 {
  width: 25%;
}
.col-xs-2 {
  width: 16.66666667%;
}
.col-xs-1 {
  width: 8.33333333%;
}
.col-xs-pull-12 {
  right: 100%;
}
.col-xs-pull-11 {
  right: 91.66666667%;
}
.col-xs-pull-10 {
  right: 83.33333333%;
}
.col-xs-pull-9 {
  right: 75%;
}
.col-xs-pull-8 {
  right: 66.66666667%;
}
.col-xs-pull-7 {
  right: 58.33333333%;
}
.col-xs-pull-6 {
  right: 50%;
}
.col-xs-pull-5 {
  right: 41.66666667%;
}
.col-xs-pull-4 {
  right: 33.33333333%;
}
.col-xs-pull-3 {
  right: 25%;
}
.col-xs-pull-2 {
  right: 16.66666667%;
}
.col-xs-pull-1 {
  right: 8.33333333%;
}
.col-xs-pull-0 {
  right: auto;
}
.col-xs-push-12 {
  left: 100%;
}
.col-xs-push-11 {
  left: 91.66666667%;
}
.col-xs-push-10 {
  left: 83.33333333%;
}
.col-xs-push-9 {
  left: 75%;
}
.col-xs-push-8 {
  left: 66.66666667%;
}
.col-xs-push-7 {
  left: 58.33333333%;
}
.col-xs-push-6 {
  left: 50%;
}
.col-xs-push-5 {
  left: 41.66666667%;
}
.col-xs-push-4 {
  left: 33.33333333%;
}
.col-xs-push-3 {
  left: 25%;
}
.col-xs-push-2 {
  left: 16.66666667%;
}
.col-xs-push-1 {
  left: 8.33333333%;
}
.col-xs-push-0 {
  left: auto;
}
.col-xs-offset-12 {
  margin-left: 100%;
}
.col-xs-offset-11 {
  margin-left: 91.66666667%;
}
.col-xs-offset-10 {
  margin-left: 83.33333333%;
}
.col-xs-offset-9 {
  margin-left: 75%;
}
.col-xs-offset-8 {
  margin-left: 66.66666667%;
}
.col-xs-offset-7 {
  margin-left: 58.33333333%;
}
.col-xs-offset-6 {
  margin-left: 50%;
}
.col-xs-offset-5 {
  margin-left: 41.66666667%;
}
.col-xs-offset-4 {
  margin-left: 33.33333333%;
}
.col-xs-offset-3 {
  margin-left: 25%;
}
.col-xs-offset-2 {
  margin-left: 16.66666667%;
}
.col-xs-offset-1 {
  margin-left: 8.33333333%;
}
.col-xs-offset-0 {
  margin-left: 0%;
}
@media (min-width: 768px) {
  .col-sm-1, .col-sm-2, .col-sm-3, .col-sm-4, .col-sm-5, .col-sm-6, .col-sm-7, .col-sm-8, .col-sm-9, .col-sm-10, .col-sm-11, .col-sm-12 {
    float: left;
  }
  .col-sm-12 {
    width: 100%;
  }
  .col-sm-11 {
    width: 91.66666667%;
  }
  .col-sm-10 {
    width: 83.33333333%;
  }
  .col-sm-9 {
    width: 75%;
  }
  .col-sm-8 {
    width: 66.66666667%;
  }
  .col-sm-7 {
    width: 58.33333333%;
  }
  .col-sm-6 {
    width: 50%;
  }
  .col-sm-5 {
    width: 41.66666667%;
  }
  .col-sm-4 {
    width: 33.33333333%;
  }
  .col-sm-3 {
    width: 25%;
  }
  .col-sm-2 {
    width: 16.66666667%;
  }
  .col-sm-1 {
    width: 8.33333333%;
  }
  .col-sm-pull-12 {
    right: 100%;
  }
  .col-sm-pull-11 {
    right: 91.66666667%;
  }
  .col-sm-pull-10 {
    right: 83.33333333%;
  }
  .col-sm-pull-9 {
    right: 75%;
  }
  .col-sm-pull-8 {
    right: 66.66666667%;
  }
  .col-sm-pull-7 {
    right: 58.33333333%;
  }
  .col-sm-pull-6 {
    right: 50%;
  }
  .col-sm-pull-5 {
    right: 41.66666667%;
  }
  .col-sm-pull-4 {
    right: 33.33333333%;
  }
  .col-sm-pull-3 {
    right: 25%;
  }
  .col-sm-pull-2 {
    right: 16.66666667%;
  }
  .col-sm-pull-1 {
    right: 8.33333333%;
  }
  .col-sm-pull-0 {
    right: auto;
  }
  .col-sm-push-12 {
    left: 100%;
  }
  .col-sm-push-11 {
    left: 91.66666667%;
  }
  .col-sm-push-10 {
    left: 83.33333333%;
  }
  .col-sm-push-9 {
    left: 75%;
  }
  .col-sm-push-8 {
    left: 66.66666667%;
  }
  .col-sm-push-7 {
    left: 58.33333333%;
  }
  .col-sm-push-6 {
    left: 50%;
  }
  .col-sm-push-5 {
    left: 41.66666667%;
  }
  .col-sm-push-4 {
    left: 33.33333333%;
  }
  .col-sm-push-3 {
    left: 25%;
  }
  .col-sm-push-2 {
    left: 16.66666667%;
  }
  .col-sm-push-1 {
    left: 8.33333333%;
  }
  .col-sm-push-0 {
    left: auto;
  }
  .col-sm-offset-12 {
    margin-left: 100%;
  }
  .col-sm-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-sm-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-sm-offset-9 {
    margin-left: 75%;
  }
  .col-sm-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-sm-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-sm-offset-6 {
    margin-left: 50%;
  }
  .col-sm-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-sm-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-sm-offset-3 {
    margin-left: 25%;
  }
  .col-sm-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-sm-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-sm-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 992px) {
  .col-md-1, .col-md-2, .col-md-3, .col-md-4, .col-md-5, .col-md-6, .col-md-7, .col-md-8, .col-md-9, .col-md-10, .col-md-11, .col-md-12 {
    float: left;
  }
  .col-md-12 {
    width: 100%;
  }
  .col-md-11 {
    width: 91.66666667%;
  }
  .col-md-10 {
    width: 83.33333333%;
  }
  .col-md-9 {
    width: 75%;
  }
  .col-md-8 {
    width: 66.66666667%;
  }
  .col-md-7 {
    width: 58.33333333%;
  }
  .col-md-6 {
    width: 50%;
  }
  .col-md-5 {
    width: 41.66666667%;
  }
  .col-md-4 {
    width: 33.33333333%;
  }
  .col-md-3 {
    width: 25%;
  }
  .col-md-2 {
    width: 16.66666667%;
  }
  .col-md-1 {
    width: 8.33333333%;
  }
  .col-md-pull-12 {
    right: 100%;
  }
  .col-md-pull-11 {
    right: 91.66666667%;
  }
  .col-md-pull-10 {
    right: 83.33333333%;
  }
  .col-md-pull-9 {
    right: 75%;
  }
  .col-md-pull-8 {
    right: 66.66666667%;
  }
  .col-md-pull-7 {
    right: 58.33333333%;
  }
  .col-md-pull-6 {
    right: 50%;
  }
  .col-md-pull-5 {
    right: 41.66666667%;
  }
  .col-md-pull-4 {
    right: 33.33333333%;
  }
  .col-md-pull-3 {
    right: 25%;
  }
  .col-md-pull-2 {
    right: 16.66666667%;
  }
  .col-md-pull-1 {
    right: 8.33333333%;
  }
  .col-md-pull-0 {
    right: auto;
  }
  .col-md-push-12 {
    left: 100%;
  }
  .col-md-push-11 {
    left: 91.66666667%;
  }
  .col-md-push-10 {
    left: 83.33333333%;
  }
  .col-md-push-9 {
    left: 75%;
  }
  .col-md-push-8 {
    left: 66.66666667%;
  }
  .col-md-push-7 {
    left: 58.33333333%;
  }
  .col-md-push-6 {
    left: 50%;
  }
  .col-md-push-5 {
    left: 41.66666667%;
  }
  .col-md-push-4 {
    left: 33.33333333%;
  }
  .col-md-push-3 {
    left: 25%;
  }
  .col-md-push-2 {
    left: 16.66666667%;
  }
  .col-md-push-1 {
    left: 8.33333333%;
  }
  .col-md-push-0 {
    left: auto;
  }
  .col-md-offset-12 {
    margin-left: 100%;
  }
  .col-md-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-md-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-md-offset-9 {
    margin-left: 75%;
  }
  .col-md-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-md-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-md-offset-6 {
    margin-left: 50%;
  }
  .col-md-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-md-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-md-offset-3 {
    margin-left: 25%;
  }
  .col-md-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-md-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-md-offset-0 {
    margin-left: 0%;
  }
}
@media (min-width: 1200px) {
  .col-lg-1, .col-lg-2, .col-lg-3, .col-lg-4, .col-lg-5, .col-lg-6, .col-lg-7, .col-lg-8, .col-lg-9, .col-lg-10, .col-lg-11, .col-lg-12 {
    float: left;
  }
  .col-lg-12 {
    width: 100%;
  }
  .col-lg-11 {
    width: 91.66666667%;
  }
  .col-lg-10 {
    width: 83.33333333%;
  }
  .col-lg-9 {
    width: 75%;
  }
  .col-lg-8 {
    width: 66.66666667%;
  }
  .col-lg-7 {
    width: 58.33333333%;
  }
  .col-lg-6 {
    width: 50%;
  }
  .col-lg-5 {
    width: 41.66666667%;
  }
  .col-lg-4 {
    width: 33.33333333%;
  }
  .col-lg-3 {
    width: 25%;
  }
  .col-lg-2 {
    width: 16.66666667%;
  }
  .col-lg-1 {
    width: 8.33333333%;
  }
  .col-lg-pull-12 {
    right: 100%;
  }
  .col-lg-pull-11 {
    right: 91.66666667%;
  }
  .col-lg-pull-10 {
    right: 83.33333333%;
  }
  .col-lg-pull-9 {
    right: 75%;
  }
  .col-lg-pull-8 {
    right: 66.66666667%;
  }
  .col-lg-pull-7 {
    right: 58.33333333%;
  }
  .col-lg-pull-6 {
    right: 50%;
  }
  .col-lg-pull-5 {
    right: 41.66666667%;
  }
  .col-lg-pull-4 {
    right: 33.33333333%;
  }
  .col-lg-pull-3 {
    right: 25%;
  }
  .col-lg-pull-2 {
    right: 16.66666667%;
  }
  .col-lg-pull-1 {
    right: 8.33333333%;
  }
  .col-lg-pull-0 {
    right: auto;
  }
  .col-lg-push-12 {
    left: 100%;
  }
  .col-lg-push-11 {
    left: 91.66666667%;
  }
  .col-lg-push-10 {
    left: 83.33333333%;
  }
  .col-lg-push-9 {
    left: 75%;
  }
  .col-lg-push-8 {
    left: 66.66666667%;
  }
  .col-lg-push-7 {
    left: 58.33333333%;
  }
  .col-lg-push-6 {
    left: 50%;
  }
  .col-lg-push-5 {
    left: 41.66666667%;
  }
  .col-lg-push-4 {
    left: 33.33333333%;
  }
  .col-lg-push-3 {
    left: 25%;
  }
  .col-lg-push-2 {
    left: 16.66666667%;
  }
  .col-lg-push-1 {
    left: 8.33333333%;
  }
  .col-lg-push-0 {
    left: auto;
  }
  .col-lg-offset-12 {
    margin-left: 100%;
  }
  .col-lg-offset-11 {
    margin-left: 91.66666667%;
  }
  .col-lg-offset-10 {
    margin-left: 83.33333333%;
  }
  .col-lg-offset-9 {
    margin-left: 75%;
  }
  .col-lg-offset-8 {
    margin-left: 66.66666667%;
  }
  .col-lg-offset-7 {
    margin-left: 58.33333333%;
  }
  .col-lg-offset-6 {
    margin-left: 50%;
  }
  .col-lg-offset-5 {
    margin-left: 41.66666667%;
  }
  .col-lg-offset-4 {
    margin-left: 33.33333333%;
  }
  .col-lg-offset-3 {
    margin-left: 25%;
  }
  .col-lg-offset-2 {
    margin-left: 16.66666667%;
  }
  .col-lg-offset-1 {
    margin-left: 8.33333333%;
  }
  .col-lg-offset-0 {
    margin-left: 0%;
  }
}
table {
  background-color: transparent;
}
caption {
  padding-top: 8px;
  padding-bottom: 8px;
  color: #777777;
  text-align: left;
}
th {
  text-align: left;
}
.table {
  width: 100%;
  max-width: 100%;
  margin-bottom: 18px;
}
.table > thead > tr > th,
.table > tbody > tr > th,
.table > tfoot > tr > th,
.table > thead > tr > td,
.table > tbody > tr > td,
.table > tfoot > tr > td {
  padding: 8px;
  line-height: 1.42857143;
  vertical-align: top;
  border-top: 1px solid #ddd;
}
.table > thead > tr > th {
  vertical-align: bottom;
  border-bottom: 2px solid #ddd;
}
.table > caption + thead > tr:first-child > th,
.table > colgroup + thead > tr:first-child > th,
.table > thead:first-child > tr:first-child > th,
.table > caption + thead > tr:first-child > td,
.table > colgroup + thead > tr:first-child > td,
.table > thead:first-child > tr:first-child > td {
  border-top: 0;
}
.table > tbody + tbody {
  border-top: 2px solid #ddd;
}
.table .table {
  background-color: #fff;
}
.table-condensed > thead > tr > th,
.table-condensed > tbody > tr > th,
.table-condensed > tfoot > tr > th,
.table-condensed > thead > tr > td,
.table-condensed > tbody > tr > td,
.table-condensed > tfoot > tr > td {
  padding: 5px;
}
.table-bordered {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > tbody > tr > th,
.table-bordered > tfoot > tr > th,
.table-bordered > thead > tr > td,
.table-bordered > tbody > tr > td,
.table-bordered > tfoot > tr > td {
  border: 1px solid #ddd;
}
.table-bordered > thead > tr > th,
.table-bordered > thead > tr > td {
  border-bottom-width: 2px;
}
.table-striped > tbody > tr:nth-of-type(odd) {
  background-color: #f9f9f9;
}
.table-hover > tbody > tr:hover {
  background-color: #f5f5f5;
}
table col[class*="col-"] {
  position: static;
  float: none;
  display: table-column;
}
table td[class*="col-"],
table th[class*="col-"] {
  position: static;
  float: none;
  display: table-cell;
}
.table > thead > tr > td.active,
.table > tbody > tr > td.active,
.table > tfoot > tr > td.active,
.table > thead > tr > th.active,
.table > tbody > tr > th.active,
.table > tfoot > tr > th.active,
.table > thead > tr.active > td,
.table > tbody > tr.active > td,
.table > tfoot > tr.active > td,
.table > thead > tr.active > th,
.table > tbody > tr.active > th,
.table > tfoot > tr.active > th {
  background-color: #f5f5f5;
}
.table-hover > tbody > tr > td.active:hover,
.table-hover > tbody > tr > th.active:hover,
.table-hover > tbody > tr.active:hover > td,
.table-hover > tbody > tr:hover > .active,
.table-hover > tbody > tr.active:hover > th {
  background-color: #e8e8e8;
}
.table > thead > tr > td.success,
.table > tbody > tr > td.success,
.table > tfoot > tr > td.success,
.table > thead > tr > th.success,
.table > tbody > tr > th.success,
.table > tfoot > tr > th.success,
.table > thead > tr.success > td,
.table > tbody > tr.success > td,
.table > tfoot > tr.success > td,
.table > thead > tr.success > th,
.table > tbody > tr.success > th,
.table > tfoot > tr.success > th {
  background-color: #dff0d8;
}
.table-hover > tbody > tr > td.success:hover,
.table-hover > tbody > tr > th.success:hover,
.table-hover > tbody > tr.success:hover > td,
.table-hover > tbody > tr:hover > .success,
.table-hover > tbody > tr.success:hover > th {
  background-color: #d0e9c6;
}
.table > thead > tr > td.info,
.table > tbody > tr > td.info,
.table > tfoot > tr > td.info,
.table > thead > tr > th.info,
.table > tbody > tr > th.info,
.table > tfoot > tr > th.info,
.table > thead > tr.info > td,
.table > tbody > tr.info > td,
.table > tfoot > tr.info > td,
.table > thead > tr.info > th,
.table > tbody > tr.info > th,
.table > tfoot > tr.info > th {
  background-color: #d9edf7;
}
.table-hover > tbody > tr > td.info:hover,
.table-hover > tbody > tr > th.info:hover,
.table-hover > tbody > tr.info:hover > td,
.table-hover > tbody > tr:hover > .info,
.table-hover > tbody > tr.info:hover > th {
  background-color: #c4e3f3;
}
.table > thead > tr > td.warning,
.table > tbody > tr > td.warning,
.table > tfoot > tr > td.warning,
.table > thead > tr > th.warning,
.table > tbody > tr > th.warning,
.table > tfoot > tr > th.warning,
.table > thead > tr.warning > td,
.table > tbody > tr.warning > td,
.table > tfoot > tr.warning > td,
.table > thead > tr.warning > th,
.table > tbody > tr.warning > th,
.table > tfoot > tr.warning > th {
  background-color: #fcf8e3;
}
.table-hover > tbody > tr > td.warning:hover,
.table-hover > tbody > tr > th.warning:hover,
.table-hover > tbody > tr.warning:hover > td,
.table-hover > tbody > tr:hover > .warning,
.table-hover > tbody > tr.warning:hover > th {
  background-color: #faf2cc;
}
.table > thead > tr > td.danger,
.table > tbody > tr > td.danger,
.table > tfoot > tr > td.danger,
.table > thead > tr > th.danger,
.table > tbody > tr > th.danger,
.table > tfoot > tr > th.danger,
.table > thead > tr.danger > td,
.table > tbody > tr.danger > td,
.table > tfoot > tr.danger > td,
.table > thead > tr.danger > th,
.table > tbody > tr.danger > th,
.table > tfoot > tr.danger > th {
  background-color: #f2dede;
}
.table-hover > tbody > tr > td.danger:hover,
.table-hover > tbody > tr > th.danger:hover,
.table-hover > tbody > tr.danger:hover > td,
.table-hover > tbody > tr:hover > .danger,
.table-hover > tbody > tr.danger:hover > th {
  background-color: #ebcccc;
}
.table-responsive {
  overflow-x: auto;
  min-height: 0.01%;
}
@media screen and (max-width: 767px) {
  .table-responsive {
    width: 100%;
    margin-bottom: 13.5px;
    overflow-y: hidden;
    -ms-overflow-style: -ms-autohiding-scrollbar;
    border: 1px solid #ddd;
  }
  .table-responsive > .table {
    margin-bottom: 0;
  }
  .table-responsive > .table > thead > tr > th,
  .table-responsive > .table > tbody > tr > th,
  .table-responsive > .table > tfoot > tr > th,
  .table-responsive > .table > thead > tr > td,
  .table-responsive > .table > tbody > tr > td,
  .table-responsive > .table > tfoot > tr > td {
    white-space: nowrap;
  }
  .table-responsive > .table-bordered {
    border: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:first-child,
  .table-responsive > .table-bordered > tbody > tr > th:first-child,
  .table-responsive > .table-bordered > tfoot > tr > th:first-child,
  .table-responsive > .table-bordered > thead > tr > td:first-child,
  .table-responsive > .table-bordered > tbody > tr > td:first-child,
  .table-responsive > .table-bordered > tfoot > tr > td:first-child {
    border-left: 0;
  }
  .table-responsive > .table-bordered > thead > tr > th:last-child,
  .table-responsive > .table-bordered > tbody > tr > th:last-child,
  .table-responsive > .table-bordered > tfoot > tr > th:last-child,
  .table-responsive > .table-bordered > thead > tr > td:last-child,
  .table-responsive > .table-bordered > tbody > tr > td:last-child,
  .table-responsive > .table-bordered > tfoot > tr > td:last-child {
    border-right: 0;
  }
  .table-responsive > .table-bordered > tbody > tr:last-child > th,
  .table-responsive > .table-bordered > tfoot > tr:last-child > th,
  .table-responsive > .table-bordered > tbody > tr:last-child > td,
  .table-responsive > .table-bordered > tfoot > tr:last-child > td {
    border-bottom: 0;
  }
}
fieldset {
  padding: 0;
  margin: 0;
  border: 0;
  min-width: 0;
}
legend {
  display: block;
  width: 100%;
  padding: 0;
  margin-bottom: 18px;
  font-size: 19.5px;
  line-height: inherit;
  color: #333333;
  border: 0;
  border-bottom: 1px solid #e5e5e5;
}
label {
  display: inline-block;
  max-width: 100%;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="search"] {
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  box-sizing: border-box;
}
input[type="radio"],
input[type="checkbox"] {
  margin: 4px 0 0;
  margin-top: 1px \9;
  line-height: normal;
}
input[type="file"] {
  display: block;
}
input[type="range"] {
  display: block;
  width: 100%;
}
select[multiple],
select[size] {
  height: auto;
}
input[type="file"]:focus,
input[type="radio"]:focus,
input[type="checkbox"]:focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
output {
  display: block;
  padding-top: 7px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
}
.form-control {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
}
.form-control:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.form-control::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #999;
}
.form-control::-webkit-input-placeholder {
  color: #999;
}
.form-control::-ms-expand {
  border: 0;
  background-color: transparent;
}
.form-control[disabled],
.form-control[readonly],
fieldset[disabled] .form-control {
  background-color: #eeeeee;
  opacity: 1;
}
.form-control[disabled],
fieldset[disabled] .form-control {
  cursor: not-allowed;
}
textarea.form-control {
  height: auto;
}
input[type="search"] {
  -webkit-appearance: none;
}
@media screen and (-webkit-min-device-pixel-ratio: 0) {
  input[type="date"].form-control,
  input[type="time"].form-control,
  input[type="datetime-local"].form-control,
  input[type="month"].form-control {
    line-height: 32px;
  }
  input[type="date"].input-sm,
  input[type="time"].input-sm,
  input[type="datetime-local"].input-sm,
  input[type="month"].input-sm,
  .input-group-sm input[type="date"],
  .input-group-sm input[type="time"],
  .input-group-sm input[type="datetime-local"],
  .input-group-sm input[type="month"] {
    line-height: 30px;
  }
  input[type="date"].input-lg,
  input[type="time"].input-lg,
  input[type="datetime-local"].input-lg,
  input[type="month"].input-lg,
  .input-group-lg input[type="date"],
  .input-group-lg input[type="time"],
  .input-group-lg input[type="datetime-local"],
  .input-group-lg input[type="month"] {
    line-height: 45px;
  }
}
.form-group {
  margin-bottom: 15px;
}
.radio,
.checkbox {
  position: relative;
  display: block;
  margin-top: 10px;
  margin-bottom: 10px;
}
.radio label,
.checkbox label {
  min-height: 18px;
  padding-left: 20px;
  margin-bottom: 0;
  font-weight: normal;
  cursor: pointer;
}
.radio input[type="radio"],
.radio-inline input[type="radio"],
.checkbox input[type="checkbox"],
.checkbox-inline input[type="checkbox"] {
  position: absolute;
  margin-left: -20px;
  margin-top: 4px \9;
}
.radio + .radio,
.checkbox + .checkbox {
  margin-top: -5px;
}
.radio-inline,
.checkbox-inline {
  position: relative;
  display: inline-block;
  padding-left: 20px;
  margin-bottom: 0;
  vertical-align: middle;
  font-weight: normal;
  cursor: pointer;
}
.radio-inline + .radio-inline,
.checkbox-inline + .checkbox-inline {
  margin-top: 0;
  margin-left: 10px;
}
input[type="radio"][disabled],
input[type="checkbox"][disabled],
input[type="radio"].disabled,
input[type="checkbox"].disabled,
fieldset[disabled] input[type="radio"],
fieldset[disabled] input[type="checkbox"] {
  cursor: not-allowed;
}
.radio-inline.disabled,
.checkbox-inline.disabled,
fieldset[disabled] .radio-inline,
fieldset[disabled] .checkbox-inline {
  cursor: not-allowed;
}
.radio.disabled label,
.checkbox.disabled label,
fieldset[disabled] .radio label,
fieldset[disabled] .checkbox label {
  cursor: not-allowed;
}
.form-control-static {
  padding-top: 7px;
  padding-bottom: 7px;
  margin-bottom: 0;
  min-height: 31px;
}
.form-control-static.input-lg,
.form-control-static.input-sm {
  padding-left: 0;
  padding-right: 0;
}
.input-sm {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-sm {
  height: 30px;
  line-height: 30px;
}
textarea.input-sm,
select[multiple].input-sm {
  height: auto;
}
.form-group-sm .form-control {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.form-group-sm select.form-control {
  height: 30px;
  line-height: 30px;
}
.form-group-sm textarea.form-control,
.form-group-sm select[multiple].form-control {
  height: auto;
}
.form-group-sm .form-control-static {
  height: 30px;
  min-height: 30px;
  padding: 6px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.input-lg {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-lg {
  height: 45px;
  line-height: 45px;
}
textarea.input-lg,
select[multiple].input-lg {
  height: auto;
}
.form-group-lg .form-control {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.form-group-lg select.form-control {
  height: 45px;
  line-height: 45px;
}
.form-group-lg textarea.form-control,
.form-group-lg select[multiple].form-control {
  height: auto;
}
.form-group-lg .form-control-static {
  height: 45px;
  min-height: 35px;
  padding: 11px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.has-feedback {
  position: relative;
}
.has-feedback .form-control {
  padding-right: 40px;
}
.form-control-feedback {
  position: absolute;
  top: 0;
  right: 0;
  z-index: 2;
  display: block;
  width: 32px;
  height: 32px;
  line-height: 32px;
  text-align: center;
  pointer-events: none;
}
.input-lg + .form-control-feedback,
.input-group-lg + .form-control-feedback,
.form-group-lg .form-control + .form-control-feedback {
  width: 45px;
  height: 45px;
  line-height: 45px;
}
.input-sm + .form-control-feedback,
.input-group-sm + .form-control-feedback,
.form-group-sm .form-control + .form-control-feedback {
  width: 30px;
  height: 30px;
  line-height: 30px;
}
.has-success .help-block,
.has-success .control-label,
.has-success .radio,
.has-success .checkbox,
.has-success .radio-inline,
.has-success .checkbox-inline,
.has-success.radio label,
.has-success.checkbox label,
.has-success.radio-inline label,
.has-success.checkbox-inline label {
  color: #3c763d;
}
.has-success .form-control {
  border-color: #3c763d;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-success .form-control:focus {
  border-color: #2b542c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #67b168;
}
.has-success .input-group-addon {
  color: #3c763d;
  border-color: #3c763d;
  background-color: #dff0d8;
}
.has-success .form-control-feedback {
  color: #3c763d;
}
.has-warning .help-block,
.has-warning .control-label,
.has-warning .radio,
.has-warning .checkbox,
.has-warning .radio-inline,
.has-warning .checkbox-inline,
.has-warning.radio label,
.has-warning.checkbox label,
.has-warning.radio-inline label,
.has-warning.checkbox-inline label {
  color: #8a6d3b;
}
.has-warning .form-control {
  border-color: #8a6d3b;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-warning .form-control:focus {
  border-color: #66512c;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #c0a16b;
}
.has-warning .input-group-addon {
  color: #8a6d3b;
  border-color: #8a6d3b;
  background-color: #fcf8e3;
}
.has-warning .form-control-feedback {
  color: #8a6d3b;
}
.has-error .help-block,
.has-error .control-label,
.has-error .radio,
.has-error .checkbox,
.has-error .radio-inline,
.has-error .checkbox-inline,
.has-error.radio label,
.has-error.checkbox label,
.has-error.radio-inline label,
.has-error.checkbox-inline label {
  color: #a94442;
}
.has-error .form-control {
  border-color: #a94442;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
}
.has-error .form-control:focus {
  border-color: #843534;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075), 0 0 6px #ce8483;
}
.has-error .input-group-addon {
  color: #a94442;
  border-color: #a94442;
  background-color: #f2dede;
}
.has-error .form-control-feedback {
  color: #a94442;
}
.has-feedback label ~ .form-control-feedback {
  top: 23px;
}
.has-feedback label.sr-only ~ .form-control-feedback {
  top: 0;
}
.help-block {
  display: block;
  margin-top: 5px;
  margin-bottom: 10px;
  color: #404040;
}
@media (min-width: 768px) {
  .form-inline .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-static {
    display: inline-block;
  }
  .form-inline .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .form-inline .input-group .input-group-addon,
  .form-inline .input-group .input-group-btn,
  .form-inline .input-group .form-control {
    width: auto;
  }
  .form-inline .input-group > .form-control {
    width: 100%;
  }
  .form-inline .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio,
  .form-inline .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .form-inline .radio label,
  .form-inline .checkbox label {
    padding-left: 0;
  }
  .form-inline .radio input[type="radio"],
  .form-inline .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .form-inline .has-feedback .form-control-feedback {
    top: 0;
  }
}
.form-horizontal .radio,
.form-horizontal .checkbox,
.form-horizontal .radio-inline,
.form-horizontal .checkbox-inline {
  margin-top: 0;
  margin-bottom: 0;
  padding-top: 7px;
}
.form-horizontal .radio,
.form-horizontal .checkbox {
  min-height: 25px;
}
.form-horizontal .form-group {
  margin-left: 0px;
  margin-right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .control-label {
    text-align: right;
    margin-bottom: 0;
    padding-top: 7px;
  }
}
.form-horizontal .has-feedback .form-control-feedback {
  right: 0px;
}
@media (min-width: 768px) {
  .form-horizontal .form-group-lg .control-label {
    padding-top: 11px;
    font-size: 17px;
  }
}
@media (min-width: 768px) {
  .form-horizontal .form-group-sm .control-label {
    padding-top: 6px;
    font-size: 12px;
  }
}
.btn {
  display: inline-block;
  margin-bottom: 0;
  font-weight: normal;
  text-align: center;
  vertical-align: middle;
  touch-action: manipulation;
  cursor: pointer;
  background-image: none;
  border: 1px solid transparent;
  white-space: nowrap;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  border-radius: 2px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}
.btn:focus,
.btn:active:focus,
.btn.active:focus,
.btn.focus,
.btn:active.focus,
.btn.active.focus {
  outline: 5px auto -webkit-focus-ring-color;
  outline-offset: -2px;
}
.btn:hover,
.btn:focus,
.btn.focus {
  color: #333;
  text-decoration: none;
}
.btn:active,
.btn.active {
  outline: 0;
  background-image: none;
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn.disabled,
.btn[disabled],
fieldset[disabled] .btn {
  cursor: not-allowed;
  opacity: 0.65;
  filter: alpha(opacity=65);
  -webkit-box-shadow: none;
  box-shadow: none;
}
a.btn.disabled,
fieldset[disabled] a.btn {
  pointer-events: none;
}
.btn-default {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.btn-default:focus,
.btn-default.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.btn-default:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.btn-default:active:hover,
.btn-default.active:hover,
.open > .dropdown-toggle.btn-default:hover,
.btn-default:active:focus,
.btn-default.active:focus,
.open > .dropdown-toggle.btn-default:focus,
.btn-default:active.focus,
.btn-default.active.focus,
.open > .dropdown-toggle.btn-default.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.btn-default:active,
.btn-default.active,
.open > .dropdown-toggle.btn-default {
  background-image: none;
}
.btn-default.disabled:hover,
.btn-default[disabled]:hover,
fieldset[disabled] .btn-default:hover,
.btn-default.disabled:focus,
.btn-default[disabled]:focus,
fieldset[disabled] .btn-default:focus,
.btn-default.disabled.focus,
.btn-default[disabled].focus,
fieldset[disabled] .btn-default.focus {
  background-color: #fff;
  border-color: #ccc;
}
.btn-default .badge {
  color: #fff;
  background-color: #333;
}
.btn-primary {
  color: #fff;
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary:focus,
.btn-primary.focus {
  color: #fff;
  background-color: #286090;
  border-color: #122b40;
}
.btn-primary:hover {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  color: #fff;
  background-color: #286090;
  border-color: #204d74;
}
.btn-primary:active:hover,
.btn-primary.active:hover,
.open > .dropdown-toggle.btn-primary:hover,
.btn-primary:active:focus,
.btn-primary.active:focus,
.open > .dropdown-toggle.btn-primary:focus,
.btn-primary:active.focus,
.btn-primary.active.focus,
.open > .dropdown-toggle.btn-primary.focus {
  color: #fff;
  background-color: #204d74;
  border-color: #122b40;
}
.btn-primary:active,
.btn-primary.active,
.open > .dropdown-toggle.btn-primary {
  background-image: none;
}
.btn-primary.disabled:hover,
.btn-primary[disabled]:hover,
fieldset[disabled] .btn-primary:hover,
.btn-primary.disabled:focus,
.btn-primary[disabled]:focus,
fieldset[disabled] .btn-primary:focus,
.btn-primary.disabled.focus,
.btn-primary[disabled].focus,
fieldset[disabled] .btn-primary.focus {
  background-color: #337ab7;
  border-color: #2e6da4;
}
.btn-primary .badge {
  color: #337ab7;
  background-color: #fff;
}
.btn-success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success:focus,
.btn-success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.btn-success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.btn-success:active:hover,
.btn-success.active:hover,
.open > .dropdown-toggle.btn-success:hover,
.btn-success:active:focus,
.btn-success.active:focus,
.open > .dropdown-toggle.btn-success:focus,
.btn-success:active.focus,
.btn-success.active.focus,
.open > .dropdown-toggle.btn-success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.btn-success:active,
.btn-success.active,
.open > .dropdown-toggle.btn-success {
  background-image: none;
}
.btn-success.disabled:hover,
.btn-success[disabled]:hover,
fieldset[disabled] .btn-success:hover,
.btn-success.disabled:focus,
.btn-success[disabled]:focus,
fieldset[disabled] .btn-success:focus,
.btn-success.disabled.focus,
.btn-success[disabled].focus,
fieldset[disabled] .btn-success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.btn-success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.btn-info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info:focus,
.btn-info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.btn-info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.btn-info:active:hover,
.btn-info.active:hover,
.open > .dropdown-toggle.btn-info:hover,
.btn-info:active:focus,
.btn-info.active:focus,
.open > .dropdown-toggle.btn-info:focus,
.btn-info:active.focus,
.btn-info.active.focus,
.open > .dropdown-toggle.btn-info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.btn-info:active,
.btn-info.active,
.open > .dropdown-toggle.btn-info {
  background-image: none;
}
.btn-info.disabled:hover,
.btn-info[disabled]:hover,
fieldset[disabled] .btn-info:hover,
.btn-info.disabled:focus,
.btn-info[disabled]:focus,
fieldset[disabled] .btn-info:focus,
.btn-info.disabled.focus,
.btn-info[disabled].focus,
fieldset[disabled] .btn-info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.btn-info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.btn-warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning:focus,
.btn-warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.btn-warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.btn-warning:active:hover,
.btn-warning.active:hover,
.open > .dropdown-toggle.btn-warning:hover,
.btn-warning:active:focus,
.btn-warning.active:focus,
.open > .dropdown-toggle.btn-warning:focus,
.btn-warning:active.focus,
.btn-warning.active.focus,
.open > .dropdown-toggle.btn-warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.btn-warning:active,
.btn-warning.active,
.open > .dropdown-toggle.btn-warning {
  background-image: none;
}
.btn-warning.disabled:hover,
.btn-warning[disabled]:hover,
fieldset[disabled] .btn-warning:hover,
.btn-warning.disabled:focus,
.btn-warning[disabled]:focus,
fieldset[disabled] .btn-warning:focus,
.btn-warning.disabled.focus,
.btn-warning[disabled].focus,
fieldset[disabled] .btn-warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.btn-warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.btn-danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger:focus,
.btn-danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.btn-danger:active:hover,
.btn-danger.active:hover,
.open > .dropdown-toggle.btn-danger:hover,
.btn-danger:active:focus,
.btn-danger.active:focus,
.open > .dropdown-toggle.btn-danger:focus,
.btn-danger:active.focus,
.btn-danger.active.focus,
.open > .dropdown-toggle.btn-danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.btn-danger:active,
.btn-danger.active,
.open > .dropdown-toggle.btn-danger {
  background-image: none;
}
.btn-danger.disabled:hover,
.btn-danger[disabled]:hover,
fieldset[disabled] .btn-danger:hover,
.btn-danger.disabled:focus,
.btn-danger[disabled]:focus,
fieldset[disabled] .btn-danger:focus,
.btn-danger.disabled.focus,
.btn-danger[disabled].focus,
fieldset[disabled] .btn-danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.btn-danger .badge {
  color: #d9534f;
  background-color: #fff;
}
.btn-link {
  color: #337ab7;
  font-weight: normal;
  border-radius: 0;
}
.btn-link,
.btn-link:active,
.btn-link.active,
.btn-link[disabled],
fieldset[disabled] .btn-link {
  background-color: transparent;
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn-link,
.btn-link:hover,
.btn-link:focus,
.btn-link:active {
  border-color: transparent;
}
.btn-link:hover,
.btn-link:focus {
  color: #23527c;
  text-decoration: underline;
  background-color: transparent;
}
.btn-link[disabled]:hover,
fieldset[disabled] .btn-link:hover,
.btn-link[disabled]:focus,
fieldset[disabled] .btn-link:focus {
  color: #777777;
  text-decoration: none;
}
.btn-lg,
.btn-group-lg > .btn {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
.btn-sm,
.btn-group-sm > .btn {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-xs,
.btn-group-xs > .btn {
  padding: 1px 5px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 5px;
}
input[type="submit"].btn-block,
input[type="reset"].btn-block,
input[type="button"].btn-block {
  width: 100%;
}
.fade {
  opacity: 0;
  -webkit-transition: opacity 0.15s linear;
  -o-transition: opacity 0.15s linear;
  transition: opacity 0.15s linear;
}
.fade.in {
  opacity: 1;
}
.collapse {
  display: none;
}
.collapse.in {
  display: block;
}
tr.collapse.in {
  display: table-row;
}
tbody.collapse.in {
  display: table-row-group;
}
.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  -webkit-transition-property: height, visibility;
  transition-property: height, visibility;
  -webkit-transition-duration: 0.35s;
  transition-duration: 0.35s;
  -webkit-transition-timing-function: ease;
  transition-timing-function: ease;
}
.caret {
  display: inline-block;
  width: 0;
  height: 0;
  margin-left: 2px;
  vertical-align: middle;
  border-top: 4px dashed;
  border-top: 4px solid \9;
  border-right: 4px solid transparent;
  border-left: 4px solid transparent;
}
.dropup,
.dropdown {
  position: relative;
}
.dropdown-toggle:focus {
  outline: 0;
}
.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 160px;
  padding: 5px 0;
  margin: 2px 0 0;
  list-style: none;
  font-size: 13px;
  text-align: left;
  background-color: #fff;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 2px;
  -webkit-box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.175);
  background-clip: padding-box;
}
.dropdown-menu.pull-right {
  right: 0;
  left: auto;
}
.dropdown-menu .divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.dropdown-menu > li > a {
  display: block;
  padding: 3px 20px;
  clear: both;
  font-weight: normal;
  line-height: 1.42857143;
  color: #333333;
  white-space: nowrap;
}
.dropdown-menu > li > a:hover,
.dropdown-menu > li > a:focus {
  text-decoration: none;
  color: #262626;
  background-color: #f5f5f5;
}
.dropdown-menu > .active > a,
.dropdown-menu > .active > a:hover,
.dropdown-menu > .active > a:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  background-color: #337ab7;
}
.dropdown-menu > .disabled > a,
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  color: #777777;
}
.dropdown-menu > .disabled > a:hover,
.dropdown-menu > .disabled > a:focus {
  text-decoration: none;
  background-color: transparent;
  background-image: none;
  filter: progid:DXImageTransform.Microsoft.gradient(enabled = false);
  cursor: not-allowed;
}
.open > .dropdown-menu {
  display: block;
}
.open > a {
  outline: 0;
}
.dropdown-menu-right {
  left: auto;
  right: 0;
}
.dropdown-menu-left {
  left: 0;
  right: auto;
}
.dropdown-header {
  display: block;
  padding: 3px 20px;
  font-size: 12px;
  line-height: 1.42857143;
  color: #777777;
  white-space: nowrap;
}
.dropdown-backdrop {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  top: 0;
  z-index: 990;
}
.pull-right > .dropdown-menu {
  right: 0;
  left: auto;
}
.dropup .caret,
.navbar-fixed-bottom .dropdown .caret {
  border-top: 0;
  border-bottom: 4px dashed;
  border-bottom: 4px solid \9;
  content: "";
}
.dropup .dropdown-menu,
.navbar-fixed-bottom .dropdown .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-bottom: 2px;
}
@media (min-width: 541px) {
  .navbar-right .dropdown-menu {
    left: auto;
    right: 0;
  }
  .navbar-right .dropdown-menu-left {
    left: 0;
    right: auto;
  }
}
.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-block;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  float: left;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover,
.btn-group > .btn:focus,
.btn-group-vertical > .btn:focus,
.btn-group > .btn:active,
.btn-group-vertical > .btn:active,
.btn-group > .btn.active,
.btn-group-vertical > .btn.active {
  z-index: 2;
}
.btn-group .btn + .btn,
.btn-group .btn + .btn-group,
.btn-group .btn-group + .btn,
.btn-group .btn-group + .btn-group {
  margin-left: -1px;
}
.btn-toolbar {
  margin-left: -5px;
}
.btn-toolbar .btn,
.btn-toolbar .btn-group,
.btn-toolbar .input-group {
  float: left;
}
.btn-toolbar > .btn,
.btn-toolbar > .btn-group,
.btn-toolbar > .input-group {
  margin-left: 5px;
}
.btn-group > .btn:not(:first-child):not(:last-child):not(.dropdown-toggle) {
  border-radius: 0;
}
.btn-group > .btn:first-child {
  margin-left: 0;
}
.btn-group > .btn:first-child:not(:last-child):not(.dropdown-toggle) {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn:last-child:not(:first-child),
.btn-group > .dropdown-toggle:not(:first-child) {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group > .btn-group {
  float: left;
}
.btn-group > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.btn-group > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.btn-group .dropdown-toggle:active,
.btn-group.open .dropdown-toggle {
  outline: 0;
}
.btn-group > .btn + .dropdown-toggle {
  padding-left: 8px;
  padding-right: 8px;
}
.btn-group > .btn-lg + .dropdown-toggle {
  padding-left: 12px;
  padding-right: 12px;
}
.btn-group.open .dropdown-toggle {
  -webkit-box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
  box-shadow: inset 0 3px 5px rgba(0, 0, 0, 0.125);
}
.btn-group.open .dropdown-toggle.btn-link {
  -webkit-box-shadow: none;
  box-shadow: none;
}
.btn .caret {
  margin-left: 0;
}
.btn-lg .caret {
  border-width: 5px 5px 0;
  border-bottom-width: 0;
}
.dropup .btn-lg .caret {
  border-width: 0 5px 5px;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group,
.btn-group-vertical > .btn-group > .btn {
  display: block;
  float: none;
  width: 100%;
  max-width: 100%;
}
.btn-group-vertical > .btn-group > .btn {
  float: none;
}
.btn-group-vertical > .btn + .btn,
.btn-group-vertical > .btn + .btn-group,
.btn-group-vertical > .btn-group + .btn,
.btn-group-vertical > .btn-group + .btn-group {
  margin-top: -1px;
  margin-left: 0;
}
.btn-group-vertical > .btn:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.btn-group-vertical > .btn:first-child:not(:last-child) {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:last-child:not(:first-child) {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
.btn-group-vertical > .btn-group:not(:first-child):not(:last-child) > .btn {
  border-radius: 0;
}
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .btn:last-child,
.btn-group-vertical > .btn-group:first-child:not(:last-child) > .dropdown-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn-group:last-child:not(:first-child) > .btn:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.btn-group-justified {
  display: table;
  width: 100%;
  table-layout: fixed;
  border-collapse: separate;
}
.btn-group-justified > .btn,
.btn-group-justified > .btn-group {
  float: none;
  display: table-cell;
  width: 1%;
}
.btn-group-justified > .btn-group .btn {
  width: 100%;
}
.btn-group-justified > .btn-group .dropdown-menu {
  left: auto;
}
[data-toggle="buttons"] > .btn input[type="radio"],
[data-toggle="buttons"] > .btn-group > .btn input[type="radio"],
[data-toggle="buttons"] > .btn input[type="checkbox"],
[data-toggle="buttons"] > .btn-group > .btn input[type="checkbox"] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}
.input-group {
  position: relative;
  display: table;
  border-collapse: separate;
}
.input-group[class*="col-"] {
  float: none;
  padding-left: 0;
  padding-right: 0;
}
.input-group .form-control {
  position: relative;
  z-index: 2;
  float: left;
  width: 100%;
  margin-bottom: 0;
}
.input-group .form-control:focus {
  z-index: 3;
}
.input-group-lg > .form-control,
.input-group-lg > .input-group-addon,
.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
  border-radius: 3px;
}
select.input-group-lg > .form-control,
select.input-group-lg > .input-group-addon,
select.input-group-lg > .input-group-btn > .btn {
  height: 45px;
  line-height: 45px;
}
textarea.input-group-lg > .form-control,
textarea.input-group-lg > .input-group-addon,
textarea.input-group-lg > .input-group-btn > .btn,
select[multiple].input-group-lg > .form-control,
select[multiple].input-group-lg > .input-group-addon,
select[multiple].input-group-lg > .input-group-btn > .btn {
  height: auto;
}
.input-group-sm > .form-control,
.input-group-sm > .input-group-addon,
.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
}
select.input-group-sm > .form-control,
select.input-group-sm > .input-group-addon,
select.input-group-sm > .input-group-btn > .btn {
  height: 30px;
  line-height: 30px;
}
textarea.input-group-sm > .form-control,
textarea.input-group-sm > .input-group-addon,
textarea.input-group-sm > .input-group-btn > .btn,
select[multiple].input-group-sm > .form-control,
select[multiple].input-group-sm > .input-group-addon,
select[multiple].input-group-sm > .input-group-btn > .btn {
  height: auto;
}
.input-group-addon,
.input-group-btn,
.input-group .form-control {
  display: table-cell;
}
.input-group-addon:not(:first-child):not(:last-child),
.input-group-btn:not(:first-child):not(:last-child),
.input-group .form-control:not(:first-child):not(:last-child) {
  border-radius: 0;
}
.input-group-addon,
.input-group-btn {
  width: 1%;
  white-space: nowrap;
  vertical-align: middle;
}
.input-group-addon {
  padding: 6px 12px;
  font-size: 13px;
  font-weight: normal;
  line-height: 1;
  color: #555555;
  text-align: center;
  background-color: #eeeeee;
  border: 1px solid #ccc;
  border-radius: 2px;
}
.input-group-addon.input-sm {
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 1px;
}
.input-group-addon.input-lg {
  padding: 10px 16px;
  font-size: 17px;
  border-radius: 3px;
}
.input-group-addon input[type="radio"],
.input-group-addon input[type="checkbox"] {
  margin-top: 0;
}
.input-group .form-control:first-child,
.input-group-addon:first-child,
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group > .btn,
.input-group-btn:first-child > .dropdown-toggle,
.input-group-btn:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group-btn:last-child > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-top-right-radius: 0;
}
.input-group-addon:first-child {
  border-right: 0;
}
.input-group .form-control:last-child,
.input-group-addon:last-child,
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group > .btn,
.input-group-btn:last-child > .dropdown-toggle,
.input-group-btn:first-child > .btn:not(:first-child),
.input-group-btn:first-child > .btn-group:not(:first-child) > .btn {
  border-bottom-left-radius: 0;
  border-top-left-radius: 0;
}
.input-group-addon:last-child {
  border-left: 0;
}
.input-group-btn {
  position: relative;
  font-size: 0;
  white-space: nowrap;
}
.input-group-btn > .btn {
  position: relative;
}
.input-group-btn > .btn + .btn {
  margin-left: -1px;
}
.input-group-btn > .btn:hover,
.input-group-btn > .btn:focus,
.input-group-btn > .btn:active {
  z-index: 2;
}
.input-group-btn:first-child > .btn,
.input-group-btn:first-child > .btn-group {
  margin-right: -1px;
}
.input-group-btn:last-child > .btn,
.input-group-btn:last-child > .btn-group {
  z-index: 2;
  margin-left: -1px;
}
.nav {
  margin-bottom: 0;
  padding-left: 0;
  list-style: none;
}
.nav > li {
  position: relative;
  display: block;
}
.nav > li > a {
  position: relative;
  display: block;
  padding: 10px 15px;
}
.nav > li > a:hover,
.nav > li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.nav > li.disabled > a {
  color: #777777;
}
.nav > li.disabled > a:hover,
.nav > li.disabled > a:focus {
  color: #777777;
  text-decoration: none;
  background-color: transparent;
  cursor: not-allowed;
}
.nav .open > a,
.nav .open > a:hover,
.nav .open > a:focus {
  background-color: #eeeeee;
  border-color: #337ab7;
}
.nav .nav-divider {
  height: 1px;
  margin: 8px 0;
  overflow: hidden;
  background-color: #e5e5e5;
}
.nav > li > a > img {
  max-width: none;
}
.nav-tabs {
  border-bottom: 1px solid #ddd;
}
.nav-tabs > li {
  float: left;
  margin-bottom: -1px;
}
.nav-tabs > li > a {
  margin-right: 2px;
  line-height: 1.42857143;
  border: 1px solid transparent;
  border-radius: 2px 2px 0 0;
}
.nav-tabs > li > a:hover {
  border-color: #eeeeee #eeeeee #ddd;
}
.nav-tabs > li.active > a,
.nav-tabs > li.active > a:hover,
.nav-tabs > li.active > a:focus {
  color: #555555;
  background-color: #fff;
  border: 1px solid #ddd;
  border-bottom-color: transparent;
  cursor: default;
}
.nav-tabs.nav-justified {
  width: 100%;
  border-bottom: 0;
}
.nav-tabs.nav-justified > li {
  float: none;
}
.nav-tabs.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-tabs.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-tabs.nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs.nav-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs.nav-justified > .active > a,
.nav-tabs.nav-justified > .active > a:hover,
.nav-tabs.nav-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs.nav-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs.nav-justified > .active > a,
  .nav-tabs.nav-justified > .active > a:hover,
  .nav-tabs.nav-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.nav-pills > li {
  float: left;
}
.nav-pills > li > a {
  border-radius: 2px;
}
.nav-pills > li + li {
  margin-left: 2px;
}
.nav-pills > li.active > a,
.nav-pills > li.active > a:hover,
.nav-pills > li.active > a:focus {
  color: #fff;
  background-color: #337ab7;
}
.nav-stacked > li {
  float: none;
}
.nav-stacked > li + li {
  margin-top: 2px;
  margin-left: 0;
}
.nav-justified {
  width: 100%;
}
.nav-justified > li {
  float: none;
}
.nav-justified > li > a {
  text-align: center;
  margin-bottom: 5px;
}
.nav-justified > .dropdown .dropdown-menu {
  top: auto;
  left: auto;
}
@media (min-width: 768px) {
  .nav-justified > li {
    display: table-cell;
    width: 1%;
  }
  .nav-justified > li > a {
    margin-bottom: 0;
  }
}
.nav-tabs-justified {
  border-bottom: 0;
}
.nav-tabs-justified > li > a {
  margin-right: 0;
  border-radius: 2px;
}
.nav-tabs-justified > .active > a,
.nav-tabs-justified > .active > a:hover,
.nav-tabs-justified > .active > a:focus {
  border: 1px solid #ddd;
}
@media (min-width: 768px) {
  .nav-tabs-justified > li > a {
    border-bottom: 1px solid #ddd;
    border-radius: 2px 2px 0 0;
  }
  .nav-tabs-justified > .active > a,
  .nav-tabs-justified > .active > a:hover,
  .nav-tabs-justified > .active > a:focus {
    border-bottom-color: #fff;
  }
}
.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar {
  position: relative;
  min-height: 30px;
  margin-bottom: 18px;
  border: 1px solid transparent;
}
@media (min-width: 541px) {
  .navbar {
    border-radius: 2px;
  }
}
@media (min-width: 541px) {
  .navbar-header {
    float: left;
  }
}
.navbar-collapse {
  overflow-x: visible;
  padding-right: 0px;
  padding-left: 0px;
  border-top: 1px solid transparent;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1);
  -webkit-overflow-scrolling: touch;
}
.navbar-collapse.in {
  overflow-y: auto;
}
@media (min-width: 541px) {
  .navbar-collapse {
    width: auto;
    border-top: 0;
    box-shadow: none;
  }
  .navbar-collapse.collapse {
    display: block !important;
    height: auto !important;
    padding-bottom: 0;
    overflow: visible !important;
  }
  .navbar-collapse.in {
    overflow-y: visible;
  }
  .navbar-fixed-top .navbar-collapse,
  .navbar-static-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    padding-left: 0;
    padding-right: 0;
  }
}
.navbar-fixed-top .navbar-collapse,
.navbar-fixed-bottom .navbar-collapse {
  max-height: 340px;
}
@media (max-device-width: 540px) and (orientation: landscape) {
  .navbar-fixed-top .navbar-collapse,
  .navbar-fixed-bottom .navbar-collapse {
    max-height: 200px;
  }
}
.container > .navbar-header,
.container-fluid > .navbar-header,
.container > .navbar-collapse,
.container-fluid > .navbar-collapse {
  margin-right: 0px;
  margin-left: 0px;
}
@media (min-width: 541px) {
  .container > .navbar-header,
  .container-fluid > .navbar-header,
  .container > .navbar-collapse,
  .container-fluid > .navbar-collapse {
    margin-right: 0;
    margin-left: 0;
  }
}
.navbar-static-top {
  z-index: 1000;
  border-width: 0 0 1px;
}
@media (min-width: 541px) {
  .navbar-static-top {
    border-radius: 0;
  }
}
.navbar-fixed-top,
.navbar-fixed-bottom {
  position: fixed;
  right: 0;
  left: 0;
  z-index: 1030;
}
@media (min-width: 541px) {
  .navbar-fixed-top,
  .navbar-fixed-bottom {
    border-radius: 0;
  }
}
.navbar-fixed-top {
  top: 0;
  border-width: 0 0 1px;
}
.navbar-fixed-bottom {
  bottom: 0;
  margin-bottom: 0;
  border-width: 1px 0 0;
}
.navbar-brand {
  float: left;
  padding: 6px 0px;
  font-size: 17px;
  line-height: 18px;
  height: 30px;
}
.navbar-brand:hover,
.navbar-brand:focus {
  text-decoration: none;
}
.navbar-brand > img {
  display: block;
}
@media (min-width: 541px) {
  .navbar > .container .navbar-brand,
  .navbar > .container-fluid .navbar-brand {
    margin-left: 0px;
  }
}
.navbar-toggle {
  position: relative;
  float: right;
  margin-right: 0px;
  padding: 9px 10px;
  margin-top: -2px;
  margin-bottom: -2px;
  background-color: transparent;
  background-image: none;
  border: 1px solid transparent;
  border-radius: 2px;
}
.navbar-toggle:focus {
  outline: 0;
}
.navbar-toggle .icon-bar {
  display: block;
  width: 22px;
  height: 2px;
  border-radius: 1px;
}
.navbar-toggle .icon-bar + .icon-bar {
  margin-top: 4px;
}
@media (min-width: 541px) {
  .navbar-toggle {
    display: none;
  }
}
.navbar-nav {
  margin: 3px 0px;
}
.navbar-nav > li > a {
  padding-top: 10px;
  padding-bottom: 10px;
  line-height: 18px;
}
@media (max-width: 540px) {
  .navbar-nav .open .dropdown-menu {
    position: static;
    float: none;
    width: auto;
    margin-top: 0;
    background-color: transparent;
    border: 0;
    box-shadow: none;
  }
  .navbar-nav .open .dropdown-menu > li > a,
  .navbar-nav .open .dropdown-menu .dropdown-header {
    padding: 5px 15px 5px 25px;
  }
  .navbar-nav .open .dropdown-menu > li > a {
    line-height: 18px;
  }
  .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-nav .open .dropdown-menu > li > a:focus {
    background-image: none;
  }
}
@media (min-width: 541px) {
  .navbar-nav {
    float: left;
    margin: 0;
  }
  .navbar-nav > li {
    float: left;
  }
  .navbar-nav > li > a {
    padding-top: 6px;
    padding-bottom: 6px;
  }
}
.navbar-form {
  margin-left: 0px;
  margin-right: 0px;
  padding: 10px 0px;
  border-top: 1px solid transparent;
  border-bottom: 1px solid transparent;
  -webkit-box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.1), 0 1px 0 rgba(255, 255, 255, 0.1);
  margin-top: -1px;
  margin-bottom: -1px;
}
@media (min-width: 768px) {
  .navbar-form .form-group {
    display: inline-block;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .navbar-form .form-control-static {
    display: inline-block;
  }
  .navbar-form .input-group {
    display: inline-table;
    vertical-align: middle;
  }
  .navbar-form .input-group .input-group-addon,
  .navbar-form .input-group .input-group-btn,
  .navbar-form .input-group .form-control {
    width: auto;
  }
  .navbar-form .input-group > .form-control {
    width: 100%;
  }
  .navbar-form .control-label {
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio,
  .navbar-form .checkbox {
    display: inline-block;
    margin-top: 0;
    margin-bottom: 0;
    vertical-align: middle;
  }
  .navbar-form .radio label,
  .navbar-form .checkbox label {
    padding-left: 0;
  }
  .navbar-form .radio input[type="radio"],
  .navbar-form .checkbox input[type="checkbox"] {
    position: relative;
    margin-left: 0;
  }
  .navbar-form .has-feedback .form-control-feedback {
    top: 0;
  }
}
@media (max-width: 540px) {
  .navbar-form .form-group {
    margin-bottom: 5px;
  }
  .navbar-form .form-group:last-child {
    margin-bottom: 0;
  }
}
@media (min-width: 541px) {
  .navbar-form {
    width: auto;
    border: 0;
    margin-left: 0;
    margin-right: 0;
    padding-top: 0;
    padding-bottom: 0;
    -webkit-box-shadow: none;
    box-shadow: none;
  }
}
.navbar-nav > li > .dropdown-menu {
  margin-top: 0;
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.navbar-fixed-bottom .navbar-nav > li > .dropdown-menu {
  margin-bottom: 0;
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.navbar-btn {
  margin-top: -1px;
  margin-bottom: -1px;
}
.navbar-btn.btn-sm {
  margin-top: 0px;
  margin-bottom: 0px;
}
.navbar-btn.btn-xs {
  margin-top: 4px;
  margin-bottom: 4px;
}
.navbar-text {
  margin-top: 6px;
  margin-bottom: 6px;
}
@media (min-width: 541px) {
  .navbar-text {
    float: left;
    margin-left: 0px;
    margin-right: 0px;
  }
}
@media (min-width: 541px) {
  .navbar-left {
    float: left !important;
    float: left;
  }
  .navbar-right {
    float: right !important;
    float: right;
    margin-right: 0px;
  }
  .navbar-right ~ .navbar-right {
    margin-right: 0;
  }
}
.navbar-default {
  background-color: #f8f8f8;
  border-color: #e7e7e7;
}
.navbar-default .navbar-brand {
  color: #777;
}
.navbar-default .navbar-brand:hover,
.navbar-default .navbar-brand:focus {
  color: #5e5e5e;
  background-color: transparent;
}
.navbar-default .navbar-text {
  color: #777;
}
.navbar-default .navbar-nav > li > a {
  color: #777;
}
.navbar-default .navbar-nav > li > a:hover,
.navbar-default .navbar-nav > li > a:focus {
  color: #333;
  background-color: transparent;
}
.navbar-default .navbar-nav > .active > a,
.navbar-default .navbar-nav > .active > a:hover,
.navbar-default .navbar-nav > .active > a:focus {
  color: #555;
  background-color: #e7e7e7;
}
.navbar-default .navbar-nav > .disabled > a,
.navbar-default .navbar-nav > .disabled > a:hover,
.navbar-default .navbar-nav > .disabled > a:focus {
  color: #ccc;
  background-color: transparent;
}
.navbar-default .navbar-toggle {
  border-color: #ddd;
}
.navbar-default .navbar-toggle:hover,
.navbar-default .navbar-toggle:focus {
  background-color: #ddd;
}
.navbar-default .navbar-toggle .icon-bar {
  background-color: #888;
}
.navbar-default .navbar-collapse,
.navbar-default .navbar-form {
  border-color: #e7e7e7;
}
.navbar-default .navbar-nav > .open > a,
.navbar-default .navbar-nav > .open > a:hover,
.navbar-default .navbar-nav > .open > a:focus {
  background-color: #e7e7e7;
  color: #555;
}
@media (max-width: 540px) {
  .navbar-default .navbar-nav .open .dropdown-menu > li > a {
    color: #777;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #333;
    background-color: transparent;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #555;
    background-color: #e7e7e7;
  }
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-default .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #ccc;
    background-color: transparent;
  }
}
.navbar-default .navbar-link {
  color: #777;
}
.navbar-default .navbar-link:hover {
  color: #333;
}
.navbar-default .btn-link {
  color: #777;
}
.navbar-default .btn-link:hover,
.navbar-default .btn-link:focus {
  color: #333;
}
.navbar-default .btn-link[disabled]:hover,
fieldset[disabled] .navbar-default .btn-link:hover,
.navbar-default .btn-link[disabled]:focus,
fieldset[disabled] .navbar-default .btn-link:focus {
  color: #ccc;
}
.navbar-inverse {
  background-color: #222;
  border-color: #080808;
}
.navbar-inverse .navbar-brand {
  color: #9d9d9d;
}
.navbar-inverse .navbar-brand:hover,
.navbar-inverse .navbar-brand:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-text {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a {
  color: #9d9d9d;
}
.navbar-inverse .navbar-nav > li > a:hover,
.navbar-inverse .navbar-nav > li > a:focus {
  color: #fff;
  background-color: transparent;
}
.navbar-inverse .navbar-nav > .active > a,
.navbar-inverse .navbar-nav > .active > a:hover,
.navbar-inverse .navbar-nav > .active > a:focus {
  color: #fff;
  background-color: #080808;
}
.navbar-inverse .navbar-nav > .disabled > a,
.navbar-inverse .navbar-nav > .disabled > a:hover,
.navbar-inverse .navbar-nav > .disabled > a:focus {
  color: #444;
  background-color: transparent;
}
.navbar-inverse .navbar-toggle {
  border-color: #333;
}
.navbar-inverse .navbar-toggle:hover,
.navbar-inverse .navbar-toggle:focus {
  background-color: #333;
}
.navbar-inverse .navbar-toggle .icon-bar {
  background-color: #fff;
}
.navbar-inverse .navbar-collapse,
.navbar-inverse .navbar-form {
  border-color: #101010;
}
.navbar-inverse .navbar-nav > .open > a,
.navbar-inverse .navbar-nav > .open > a:hover,
.navbar-inverse .navbar-nav > .open > a:focus {
  background-color: #080808;
  color: #fff;
}
@media (max-width: 540px) {
  .navbar-inverse .navbar-nav .open .dropdown-menu > .dropdown-header {
    border-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu .divider {
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a {
    color: #9d9d9d;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > li > a:focus {
    color: #fff;
    background-color: transparent;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .active > a:focus {
    color: #fff;
    background-color: #080808;
  }
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:hover,
  .navbar-inverse .navbar-nav .open .dropdown-menu > .disabled > a:focus {
    color: #444;
    background-color: transparent;
  }
}
.navbar-inverse .navbar-link {
  color: #9d9d9d;
}
.navbar-inverse .navbar-link:hover {
  color: #fff;
}
.navbar-inverse .btn-link {
  color: #9d9d9d;
}
.navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link:focus {
  color: #fff;
}
.navbar-inverse .btn-link[disabled]:hover,
fieldset[disabled] .navbar-inverse .btn-link:hover,
.navbar-inverse .btn-link[disabled]:focus,
fieldset[disabled] .navbar-inverse .btn-link:focus {
  color: #444;
}
.breadcrumb {
  padding: 8px 15px;
  margin-bottom: 18px;
  list-style: none;
  background-color: #f5f5f5;
  border-radius: 2px;
}
.breadcrumb > li {
  display: inline-block;
}
.breadcrumb > li + li:before {
  content: "/\00a0";
  padding: 0 5px;
  color: #5e5e5e;
}
.breadcrumb > .active {
  color: #777777;
}
.pagination {
  display: inline-block;
  padding-left: 0;
  margin: 18px 0;
  border-radius: 2px;
}
.pagination > li {
  display: inline;
}
.pagination > li > a,
.pagination > li > span {
  position: relative;
  float: left;
  padding: 6px 12px;
  line-height: 1.42857143;
  text-decoration: none;
  color: #337ab7;
  background-color: #fff;
  border: 1px solid #ddd;
  margin-left: -1px;
}
.pagination > li:first-child > a,
.pagination > li:first-child > span {
  margin-left: 0;
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.pagination > li:last-child > a,
.pagination > li:last-child > span {
  border-bottom-right-radius: 2px;
  border-top-right-radius: 2px;
}
.pagination > li > a:hover,
.pagination > li > span:hover,
.pagination > li > a:focus,
.pagination > li > span:focus {
  z-index: 2;
  color: #23527c;
  background-color: #eeeeee;
  border-color: #ddd;
}
.pagination > .active > a,
.pagination > .active > span,
.pagination > .active > a:hover,
.pagination > .active > span:hover,
.pagination > .active > a:focus,
.pagination > .active > span:focus {
  z-index: 3;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
  cursor: default;
}
.pagination > .disabled > span,
.pagination > .disabled > span:hover,
.pagination > .disabled > span:focus,
.pagination > .disabled > a,
.pagination > .disabled > a:hover,
.pagination > .disabled > a:focus {
  color: #777777;
  background-color: #fff;
  border-color: #ddd;
  cursor: not-allowed;
}
.pagination-lg > li > a,
.pagination-lg > li > span {
  padding: 10px 16px;
  font-size: 17px;
  line-height: 1.3333333;
}
.pagination-lg > li:first-child > a,
.pagination-lg > li:first-child > span {
  border-bottom-left-radius: 3px;
  border-top-left-radius: 3px;
}
.pagination-lg > li:last-child > a,
.pagination-lg > li:last-child > span {
  border-bottom-right-radius: 3px;
  border-top-right-radius: 3px;
}
.pagination-sm > li > a,
.pagination-sm > li > span {
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
}
.pagination-sm > li:first-child > a,
.pagination-sm > li:first-child > span {
  border-bottom-left-radius: 1px;
  border-top-left-radius: 1px;
}
.pagination-sm > li:last-child > a,
.pagination-sm > li:last-child > span {
  border-bottom-right-radius: 1px;
  border-top-right-radius: 1px;
}
.pager {
  padding-left: 0;
  margin: 18px 0;
  list-style: none;
  text-align: center;
}
.pager li {
  display: inline;
}
.pager li > a,
.pager li > span {
  display: inline-block;
  padding: 5px 14px;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 15px;
}
.pager li > a:hover,
.pager li > a:focus {
  text-decoration: none;
  background-color: #eeeeee;
}
.pager .next > a,
.pager .next > span {
  float: right;
}
.pager .previous > a,
.pager .previous > span {
  float: left;
}
.pager .disabled > a,
.pager .disabled > a:hover,
.pager .disabled > a:focus,
.pager .disabled > span {
  color: #777777;
  background-color: #fff;
  cursor: not-allowed;
}
.label {
  display: inline;
  padding: .2em .6em .3em;
  font-size: 75%;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  vertical-align: baseline;
  border-radius: .25em;
}
a.label:hover,
a.label:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.label:empty {
  display: none;
}
.btn .label {
  position: relative;
  top: -1px;
}
.label-default {
  background-color: #777777;
}
.label-default[href]:hover,
.label-default[href]:focus {
  background-color: #5e5e5e;
}
.label-primary {
  background-color: #337ab7;
}
.label-primary[href]:hover,
.label-primary[href]:focus {
  background-color: #286090;
}
.label-success {
  background-color: #5cb85c;
}
.label-success[href]:hover,
.label-success[href]:focus {
  background-color: #449d44;
}
.label-info {
  background-color: #5bc0de;
}
.label-info[href]:hover,
.label-info[href]:focus {
  background-color: #31b0d5;
}
.label-warning {
  background-color: #f0ad4e;
}
.label-warning[href]:hover,
.label-warning[href]:focus {
  background-color: #ec971f;
}
.label-danger {
  background-color: #d9534f;
}
.label-danger[href]:hover,
.label-danger[href]:focus {
  background-color: #c9302c;
}
.badge {
  display: inline-block;
  min-width: 10px;
  padding: 3px 7px;
  font-size: 12px;
  font-weight: bold;
  color: #fff;
  line-height: 1;
  vertical-align: middle;
  white-space: nowrap;
  text-align: center;
  background-color: #777777;
  border-radius: 10px;
}
.badge:empty {
  display: none;
}
.btn .badge {
  position: relative;
  top: -1px;
}
.btn-xs .badge,
.btn-group-xs > .btn .badge {
  top: 0;
  padding: 1px 5px;
}
a.badge:hover,
a.badge:focus {
  color: #fff;
  text-decoration: none;
  cursor: pointer;
}
.list-group-item.active > .badge,
.nav-pills > .active > a > .badge {
  color: #337ab7;
  background-color: #fff;
}
.list-group-item > .badge {
  float: right;
}
.list-group-item > .badge + .badge {
  margin-right: 5px;
}
.nav-pills > li > a > .badge {
  margin-left: 3px;
}
.jumbotron {
  padding-top: 30px;
  padding-bottom: 30px;
  margin-bottom: 30px;
  color: inherit;
  background-color: #eeeeee;
}
.jumbotron h1,
.jumbotron .h1 {
  color: inherit;
}
.jumbotron p {
  margin-bottom: 15px;
  font-size: 20px;
  font-weight: 200;
}
.jumbotron > hr {
  border-top-color: #d5d5d5;
}
.container .jumbotron,
.container-fluid .jumbotron {
  border-radius: 3px;
  padding-left: 0px;
  padding-right: 0px;
}
.jumbotron .container {
  max-width: 100%;
}
@media screen and (min-width: 768px) {
  .jumbotron {
    padding-top: 48px;
    padding-bottom: 48px;
  }
  .container .jumbotron,
  .container-fluid .jumbotron {
    padding-left: 60px;
    padding-right: 60px;
  }
  .jumbotron h1,
  .jumbotron .h1 {
    font-size: 59px;
  }
}
.thumbnail {
  display: block;
  padding: 4px;
  margin-bottom: 18px;
  line-height: 1.42857143;
  background-color: #fff;
  border: 1px solid #ddd;
  border-radius: 2px;
  -webkit-transition: border 0.2s ease-in-out;
  -o-transition: border 0.2s ease-in-out;
  transition: border 0.2s ease-in-out;
}
.thumbnail > img,
.thumbnail a > img {
  margin-left: auto;
  margin-right: auto;
}
a.thumbnail:hover,
a.thumbnail:focus,
a.thumbnail.active {
  border-color: #337ab7;
}
.thumbnail .caption {
  padding: 9px;
  color: #000;
}
.alert {
  padding: 15px;
  margin-bottom: 18px;
  border: 1px solid transparent;
  border-radius: 2px;
}
.alert h4 {
  margin-top: 0;
  color: inherit;
}
.alert .alert-link {
  font-weight: bold;
}
.alert > p,
.alert > ul {
  margin-bottom: 0;
}
.alert > p + p {
  margin-top: 5px;
}
.alert-dismissable,
.alert-dismissible {
  padding-right: 35px;
}
.alert-dismissable .close,
.alert-dismissible .close {
  position: relative;
  top: -2px;
  right: -21px;
  color: inherit;
}
.alert-success {
  background-color: #dff0d8;
  border-color: #d6e9c6;
  color: #3c763d;
}
.alert-success hr {
  border-top-color: #c9e2b3;
}
.alert-success .alert-link {
  color: #2b542c;
}
.alert-info {
  background-color: #d9edf7;
  border-color: #bce8f1;
  color: #31708f;
}
.alert-info hr {
  border-top-color: #a6e1ec;
}
.alert-info .alert-link {
  color: #245269;
}
.alert-warning {
  background-color: #fcf8e3;
  border-color: #faebcc;
  color: #8a6d3b;
}
.alert-warning hr {
  border-top-color: #f7e1b5;
}
.alert-warning .alert-link {
  color: #66512c;
}
.alert-danger {
  background-color: #f2dede;
  border-color: #ebccd1;
  color: #a94442;
}
.alert-danger hr {
  border-top-color: #e4b9c0;
}
.alert-danger .alert-link {
  color: #843534;
}
@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
@keyframes progress-bar-stripes {
  from {
    background-position: 40px 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  overflow: hidden;
  height: 18px;
  margin-bottom: 18px;
  background-color: #f5f5f5;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.1);
}
.progress-bar {
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 18px;
  color: #fff;
  text-align: center;
  background-color: #337ab7;
  -webkit-box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  -webkit-transition: width 0.6s ease;
  -o-transition: width 0.6s ease;
  transition: width 0.6s ease;
}
.progress-striped .progress-bar,
.progress-bar-striped {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 40px 40px;
}
.progress.active .progress-bar,
.progress-bar.active {
  -webkit-animation: progress-bar-stripes 2s linear infinite;
  -o-animation: progress-bar-stripes 2s linear infinite;
  animation: progress-bar-stripes 2s linear infinite;
}
.progress-bar-success {
  background-color: #5cb85c;
}
.progress-striped .progress-bar-success {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-info {
  background-color: #5bc0de;
}
.progress-striped .progress-bar-info {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-warning {
  background-color: #f0ad4e;
}
.progress-striped .progress-bar-warning {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.progress-bar-danger {
  background-color: #d9534f;
}
.progress-striped .progress-bar-danger {
  background-image: -webkit-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: -o-linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
}
.media {
  margin-top: 15px;
}
.media:first-child {
  margin-top: 0;
}
.media,
.media-body {
  zoom: 1;
  overflow: hidden;
}
.media-body {
  width: 10000px;
}
.media-object {
  display: block;
}
.media-object.img-thumbnail {
  max-width: none;
}
.media-right,
.media > .pull-right {
  padding-left: 10px;
}
.media-left,
.media > .pull-left {
  padding-right: 10px;
}
.media-left,
.media-right,
.media-body {
  display: table-cell;
  vertical-align: top;
}
.media-middle {
  vertical-align: middle;
}
.media-bottom {
  vertical-align: bottom;
}
.media-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.media-list {
  padding-left: 0;
  list-style: none;
}
.list-group {
  margin-bottom: 20px;
  padding-left: 0;
}
.list-group-item {
  position: relative;
  display: block;
  padding: 10px 15px;
  margin-bottom: -1px;
  background-color: #fff;
  border: 1px solid #ddd;
}
.list-group-item:first-child {
  border-top-right-radius: 2px;
  border-top-left-radius: 2px;
}
.list-group-item:last-child {
  margin-bottom: 0;
  border-bottom-right-radius: 2px;
  border-bottom-left-radius: 2px;
}
a.list-group-item,
button.list-group-item {
  color: #555;
}
a.list-group-item .list-group-item-heading,
button.list-group-item .list-group-item-heading {
  color: #333;
}
a.list-group-item:hover,
button.list-group-item:hover,
a.list-group-item:focus,
button.list-group-item:focus {
  text-decoration: none;
  color: #555;
  background-color: #f5f5f5;
}
button.list-group-item {
  width: 100%;
  text-align: left;
}
.list-group-item.disabled,
.list-group-item.disabled:hover,
.list-group-item.disabled:focus {
  background-color: #eeeeee;
  color: #777777;
  cursor: not-allowed;
}
.list-group-item.disabled .list-group-item-heading,
.list-group-item.disabled:hover .list-group-item-heading,
.list-group-item.disabled:focus .list-group-item-heading {
  color: inherit;
}
.list-group-item.disabled .list-group-item-text,
.list-group-item.disabled:hover .list-group-item-text,
.list-group-item.disabled:focus .list-group-item-text {
  color: #777777;
}
.list-group-item.active,
.list-group-item.active:hover,
.list-group-item.active:focus {
  z-index: 2;
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.list-group-item.active .list-group-item-heading,
.list-group-item.active:hover .list-group-item-heading,
.list-group-item.active:focus .list-group-item-heading,
.list-group-item.active .list-group-item-heading > small,
.list-group-item.active:hover .list-group-item-heading > small,
.list-group-item.active:focus .list-group-item-heading > small,
.list-group-item.active .list-group-item-heading > .small,
.list-group-item.active:hover .list-group-item-heading > .small,
.list-group-item.active:focus .list-group-item-heading > .small {
  color: inherit;
}
.list-group-item.active .list-group-item-text,
.list-group-item.active:hover .list-group-item-text,
.list-group-item.active:focus .list-group-item-text {
  color: #c7ddef;
}
.list-group-item-success {
  color: #3c763d;
  background-color: #dff0d8;
}
a.list-group-item-success,
button.list-group-item-success {
  color: #3c763d;
}
a.list-group-item-success .list-group-item-heading,
button.list-group-item-success .list-group-item-heading {
  color: inherit;
}
a.list-group-item-success:hover,
button.list-group-item-success:hover,
a.list-group-item-success:focus,
button.list-group-item-success:focus {
  color: #3c763d;
  background-color: #d0e9c6;
}
a.list-group-item-success.active,
button.list-group-item-success.active,
a.list-group-item-success.active:hover,
button.list-group-item-success.active:hover,
a.list-group-item-success.active:focus,
button.list-group-item-success.active:focus {
  color: #fff;
  background-color: #3c763d;
  border-color: #3c763d;
}
.list-group-item-info {
  color: #31708f;
  background-color: #d9edf7;
}
a.list-group-item-info,
button.list-group-item-info {
  color: #31708f;
}
a.list-group-item-info .list-group-item-heading,
button.list-group-item-info .list-group-item-heading {
  color: inherit;
}
a.list-group-item-info:hover,
button.list-group-item-info:hover,
a.list-group-item-info:focus,
button.list-group-item-info:focus {
  color: #31708f;
  background-color: #c4e3f3;
}
a.list-group-item-info.active,
button.list-group-item-info.active,
a.list-group-item-info.active:hover,
button.list-group-item-info.active:hover,
a.list-group-item-info.active:focus,
button.list-group-item-info.active:focus {
  color: #fff;
  background-color: #31708f;
  border-color: #31708f;
}
.list-group-item-warning {
  color: #8a6d3b;
  background-color: #fcf8e3;
}
a.list-group-item-warning,
button.list-group-item-warning {
  color: #8a6d3b;
}
a.list-group-item-warning .list-group-item-heading,
button.list-group-item-warning .list-group-item-heading {
  color: inherit;
}
a.list-group-item-warning:hover,
button.list-group-item-warning:hover,
a.list-group-item-warning:focus,
button.list-group-item-warning:focus {
  color: #8a6d3b;
  background-color: #faf2cc;
}
a.list-group-item-warning.active,
button.list-group-item-warning.active,
a.list-group-item-warning.active:hover,
button.list-group-item-warning.active:hover,
a.list-group-item-warning.active:focus,
button.list-group-item-warning.active:focus {
  color: #fff;
  background-color: #8a6d3b;
  border-color: #8a6d3b;
}
.list-group-item-danger {
  color: #a94442;
  background-color: #f2dede;
}
a.list-group-item-danger,
button.list-group-item-danger {
  color: #a94442;
}
a.list-group-item-danger .list-group-item-heading,
button.list-group-item-danger .list-group-item-heading {
  color: inherit;
}
a.list-group-item-danger:hover,
button.list-group-item-danger:hover,
a.list-group-item-danger:focus,
button.list-group-item-danger:focus {
  color: #a94442;
  background-color: #ebcccc;
}
a.list-group-item-danger.active,
button.list-group-item-danger.active,
a.list-group-item-danger.active:hover,
button.list-group-item-danger.active:hover,
a.list-group-item-danger.active:focus,
button.list-group-item-danger.active:focus {
  color: #fff;
  background-color: #a94442;
  border-color: #a94442;
}
.list-group-item-heading {
  margin-top: 0;
  margin-bottom: 5px;
}
.list-group-item-text {
  margin-bottom: 0;
  line-height: 1.3;
}
.panel {
  margin-bottom: 18px;
  background-color: #fff;
  border: 1px solid transparent;
  border-radius: 2px;
  -webkit-box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);
}
.panel-body {
  padding: 15px;
}
.panel-heading {
  padding: 10px 15px;
  border-bottom: 1px solid transparent;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel-heading > .dropdown .dropdown-toggle {
  color: inherit;
}
.panel-title {
  margin-top: 0;
  margin-bottom: 0;
  font-size: 15px;
  color: inherit;
}
.panel-title > a,
.panel-title > small,
.panel-title > .small,
.panel-title > small > a,
.panel-title > .small > a {
  color: inherit;
}
.panel-footer {
  padding: 10px 15px;
  background-color: #f5f5f5;
  border-top: 1px solid #ddd;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .list-group,
.panel > .panel-collapse > .list-group {
  margin-bottom: 0;
}
.panel > .list-group .list-group-item,
.panel > .panel-collapse > .list-group .list-group-item {
  border-width: 1px 0;
  border-radius: 0;
}
.panel > .list-group:first-child .list-group-item:first-child,
.panel > .panel-collapse > .list-group:first-child .list-group-item:first-child {
  border-top: 0;
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .list-group:last-child .list-group-item:last-child,
.panel > .panel-collapse > .list-group:last-child .list-group-item:last-child {
  border-bottom: 0;
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .panel-heading + .panel-collapse > .list-group .list-group-item:first-child {
  border-top-right-radius: 0;
  border-top-left-radius: 0;
}
.panel-heading + .list-group .list-group-item:first-child {
  border-top-width: 0;
}
.list-group + .panel-footer {
  border-top-width: 0;
}
.panel > .table,
.panel > .table-responsive > .table,
.panel > .panel-collapse > .table {
  margin-bottom: 0;
}
.panel > .table caption,
.panel > .table-responsive > .table caption,
.panel > .panel-collapse > .table caption {
  padding-left: 15px;
  padding-right: 15px;
}
.panel > .table:first-child,
.panel > .table-responsive:first-child > .table:first-child {
  border-top-right-radius: 1px;
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child {
  border-top-left-radius: 1px;
  border-top-right-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:first-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:first-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:first-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:first-child {
  border-top-left-radius: 1px;
}
.panel > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child td:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child td:last-child,
.panel > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > thead:first-child > tr:first-child th:last-child,
.panel > .table:first-child > tbody:first-child > tr:first-child th:last-child,
.panel > .table-responsive:first-child > .table:first-child > tbody:first-child > tr:first-child th:last-child {
  border-top-right-radius: 1px;
}
.panel > .table:last-child,
.panel > .table-responsive:last-child > .table:last-child {
  border-bottom-right-radius: 1px;
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child {
  border-bottom-left-radius: 1px;
  border-bottom-right-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:first-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:first-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:first-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:first-child {
  border-bottom-left-radius: 1px;
}
.panel > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child td:last-child,
.panel > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tbody:last-child > tr:last-child th:last-child,
.panel > .table:last-child > tfoot:last-child > tr:last-child th:last-child,
.panel > .table-responsive:last-child > .table:last-child > tfoot:last-child > tr:last-child th:last-child {
  border-bottom-right-radius: 1px;
}
.panel > .panel-body + .table,
.panel > .panel-body + .table-responsive,
.panel > .table + .panel-body,
.panel > .table-responsive + .panel-body {
  border-top: 1px solid #ddd;
}
.panel > .table > tbody:first-child > tr:first-child th,
.panel > .table > tbody:first-child > tr:first-child td {
  border-top: 0;
}
.panel > .table-bordered,
.panel > .table-responsive > .table-bordered {
  border: 0;
}
.panel > .table-bordered > thead > tr > th:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:first-child,
.panel > .table-bordered > tbody > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:first-child,
.panel > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:first-child,
.panel > .table-bordered > thead > tr > td:first-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:first-child,
.panel > .table-bordered > tbody > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:first-child,
.panel > .table-bordered > tfoot > tr > td:first-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:first-child {
  border-left: 0;
}
.panel > .table-bordered > thead > tr > th:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > th:last-child,
.panel > .table-bordered > tbody > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > th:last-child,
.panel > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > th:last-child,
.panel > .table-bordered > thead > tr > td:last-child,
.panel > .table-responsive > .table-bordered > thead > tr > td:last-child,
.panel > .table-bordered > tbody > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tbody > tr > td:last-child,
.panel > .table-bordered > tfoot > tr > td:last-child,
.panel > .table-responsive > .table-bordered > tfoot > tr > td:last-child {
  border-right: 0;
}
.panel > .table-bordered > thead > tr:first-child > td,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > td,
.panel > .table-bordered > tbody > tr:first-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > td,
.panel > .table-bordered > thead > tr:first-child > th,
.panel > .table-responsive > .table-bordered > thead > tr:first-child > th,
.panel > .table-bordered > tbody > tr:first-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:first-child > th {
  border-bottom: 0;
}
.panel > .table-bordered > tbody > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > td,
.panel > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > td,
.panel > .table-bordered > tbody > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tbody > tr:last-child > th,
.panel > .table-bordered > tfoot > tr:last-child > th,
.panel > .table-responsive > .table-bordered > tfoot > tr:last-child > th {
  border-bottom: 0;
}
.panel > .table-responsive {
  border: 0;
  margin-bottom: 0;
}
.panel-group {
  margin-bottom: 18px;
}
.panel-group .panel {
  margin-bottom: 0;
  border-radius: 2px;
}
.panel-group .panel + .panel {
  margin-top: 5px;
}
.panel-group .panel-heading {
  border-bottom: 0;
}
.panel-group .panel-heading + .panel-collapse > .panel-body,
.panel-group .panel-heading + .panel-collapse > .list-group {
  border-top: 1px solid #ddd;
}
.panel-group .panel-footer {
  border-top: 0;
}
.panel-group .panel-footer + .panel-collapse .panel-body {
  border-bottom: 1px solid #ddd;
}
.panel-default {
  border-color: #ddd;
}
.panel-default > .panel-heading {
  color: #333333;
  background-color: #f5f5f5;
  border-color: #ddd;
}
.panel-default > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ddd;
}
.panel-default > .panel-heading .badge {
  color: #f5f5f5;
  background-color: #333333;
}
.panel-default > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ddd;
}
.panel-primary {
  border-color: #337ab7;
}
.panel-primary > .panel-heading {
  color: #fff;
  background-color: #337ab7;
  border-color: #337ab7;
}
.panel-primary > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #337ab7;
}
.panel-primary > .panel-heading .badge {
  color: #337ab7;
  background-color: #fff;
}
.panel-primary > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #337ab7;
}
.panel-success {
  border-color: #d6e9c6;
}
.panel-success > .panel-heading {
  color: #3c763d;
  background-color: #dff0d8;
  border-color: #d6e9c6;
}
.panel-success > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #d6e9c6;
}
.panel-success > .panel-heading .badge {
  color: #dff0d8;
  background-color: #3c763d;
}
.panel-success > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #d6e9c6;
}
.panel-info {
  border-color: #bce8f1;
}
.panel-info > .panel-heading {
  color: #31708f;
  background-color: #d9edf7;
  border-color: #bce8f1;
}
.panel-info > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #bce8f1;
}
.panel-info > .panel-heading .badge {
  color: #d9edf7;
  background-color: #31708f;
}
.panel-info > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #bce8f1;
}
.panel-warning {
  border-color: #faebcc;
}
.panel-warning > .panel-heading {
  color: #8a6d3b;
  background-color: #fcf8e3;
  border-color: #faebcc;
}
.panel-warning > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #faebcc;
}
.panel-warning > .panel-heading .badge {
  color: #fcf8e3;
  background-color: #8a6d3b;
}
.panel-warning > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #faebcc;
}
.panel-danger {
  border-color: #ebccd1;
}
.panel-danger > .panel-heading {
  color: #a94442;
  background-color: #f2dede;
  border-color: #ebccd1;
}
.panel-danger > .panel-heading + .panel-collapse > .panel-body {
  border-top-color: #ebccd1;
}
.panel-danger > .panel-heading .badge {
  color: #f2dede;
  background-color: #a94442;
}
.panel-danger > .panel-footer + .panel-collapse > .panel-body {
  border-bottom-color: #ebccd1;
}
.embed-responsive {
  position: relative;
  display: block;
  height: 0;
  padding: 0;
  overflow: hidden;
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  height: 100%;
  width: 100%;
  border: 0;
}
.embed-responsive-16by9 {
  padding-bottom: 56.25%;
}
.embed-responsive-4by3 {
  padding-bottom: 75%;
}
.well {
  min-height: 20px;
  padding: 19px;
  margin-bottom: 20px;
  background-color: #f5f5f5;
  border: 1px solid #e3e3e3;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.05);
}
.well blockquote {
  border-color: #ddd;
  border-color: rgba(0, 0, 0, 0.15);
}
.well-lg {
  padding: 24px;
  border-radius: 3px;
}
.well-sm {
  padding: 9px;
  border-radius: 1px;
}
.close {
  float: right;
  font-size: 19.5px;
  font-weight: bold;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.2;
  filter: alpha(opacity=20);
}
.close:hover,
.close:focus {
  color: #000;
  text-decoration: none;
  cursor: pointer;
  opacity: 0.5;
  filter: alpha(opacity=50);
}
button.close {
  padding: 0;
  cursor: pointer;
  background: transparent;
  border: 0;
  -webkit-appearance: none;
}
.modal-open {
  overflow: hidden;
}
.modal {
  display: none;
  overflow: hidden;
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1050;
  -webkit-overflow-scrolling: touch;
  outline: 0;
}
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, -25%);
  -ms-transform: translate(0, -25%);
  -o-transform: translate(0, -25%);
  transform: translate(0, -25%);
  -webkit-transition: -webkit-transform 0.3s ease-out;
  -moz-transition: -moz-transform 0.3s ease-out;
  -o-transition: -o-transform 0.3s ease-out;
  transition: transform 0.3s ease-out;
}
.modal.in .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}
.modal-dialog {
  position: relative;
  width: auto;
  margin: 10px;
}
.modal-content {
  position: relative;
  background-color: #fff;
  border: 1px solid #999;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  box-shadow: 0 3px 9px rgba(0, 0, 0, 0.5);
  background-clip: padding-box;
  outline: 0;
}
.modal-backdrop {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 1040;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
  filter: alpha(opacity=0);
}
.modal-backdrop.in {
  opacity: 0.5;
  filter: alpha(opacity=50);
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #e5e5e5;
}
.modal-header .close {
  margin-top: -2px;
}
.modal-title {
  margin: 0;
  line-height: 1.42857143;
}
.modal-body {
  position: relative;
  padding: 15px;
}
.modal-footer {
  padding: 15px;
  text-align: right;
  border-top: 1px solid #e5e5e5;
}
.modal-footer .btn + .btn {
  margin-left: 5px;
  margin-bottom: 0;
}
.modal-footer .btn-group .btn + .btn {
  margin-left: -1px;
}
.modal-footer .btn-block + .btn-block {
  margin-left: 0;
}
.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}
@media (min-width: 768px) {
  .modal-dialog {
    width: 600px;
    margin: 30px auto;
  }
  .modal-content {
    -webkit-box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.5);
  }
  .modal-sm {
    width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg {
    width: 900px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 12px;
  opacity: 0;
  filter: alpha(opacity=0);
}
.tooltip.in {
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.tooltip.top {
  margin-top: -3px;
  padding: 5px 0;
}
.tooltip.right {
  margin-left: 3px;
  padding: 0 5px;
}
.tooltip.bottom {
  margin-top: 3px;
  padding: 5px 0;
}
.tooltip.left {
  margin-left: -3px;
  padding: 0 5px;
}
.tooltip-inner {
  max-width: 200px;
  padding: 3px 8px;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 2px;
}
.tooltip-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.tooltip.top .tooltip-arrow {
  bottom: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-left .tooltip-arrow {
  bottom: 0;
  right: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.top-right .tooltip-arrow {
  bottom: 0;
  left: 5px;
  margin-bottom: -5px;
  border-width: 5px 5px 0;
  border-top-color: #000;
}
.tooltip.right .tooltip-arrow {
  top: 50%;
  left: 0;
  margin-top: -5px;
  border-width: 5px 5px 5px 0;
  border-right-color: #000;
}
.tooltip.left .tooltip-arrow {
  top: 50%;
  right: 0;
  margin-top: -5px;
  border-width: 5px 0 5px 5px;
  border-left-color: #000;
}
.tooltip.bottom .tooltip-arrow {
  top: 0;
  left: 50%;
  margin-left: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-left .tooltip-arrow {
  top: 0;
  right: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.tooltip.bottom-right .tooltip-arrow {
  top: 0;
  left: 5px;
  margin-top: -5px;
  border-width: 0 5px 5px;
  border-bottom-color: #000;
}
.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: none;
  max-width: 276px;
  padding: 1px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-style: normal;
  font-weight: normal;
  letter-spacing: normal;
  line-break: auto;
  line-height: 1.42857143;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  white-space: normal;
  word-break: normal;
  word-spacing: normal;
  word-wrap: normal;
  font-size: 13px;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ccc;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 3px;
  -webkit-box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
  box-shadow: 0 5px 10px rgba(0, 0, 0, 0.2);
}
.popover.top {
  margin-top: -10px;
}
.popover.right {
  margin-left: 10px;
}
.popover.bottom {
  margin-top: 10px;
}
.popover.left {
  margin-left: -10px;
}
.popover-title {
  margin: 0;
  padding: 8px 14px;
  font-size: 13px;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-radius: 2px 2px 0 0;
}
.popover-content {
  padding: 9px 14px;
}
.popover > .arrow,
.popover > .arrow:after {
  position: absolute;
  display: block;
  width: 0;
  height: 0;
  border-color: transparent;
  border-style: solid;
}
.popover > .arrow {
  border-width: 11px;
}
.popover > .arrow:after {
  border-width: 10px;
  content: "";
}
.popover.top > .arrow {
  left: 50%;
  margin-left: -11px;
  border-bottom-width: 0;
  border-top-color: #999999;
  border-top-color: rgba(0, 0, 0, 0.25);
  bottom: -11px;
}
.popover.top > .arrow:after {
  content: " ";
  bottom: 1px;
  margin-left: -10px;
  border-bottom-width: 0;
  border-top-color: #fff;
}
.popover.right > .arrow {
  top: 50%;
  left: -11px;
  margin-top: -11px;
  border-left-width: 0;
  border-right-color: #999999;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.popover.right > .arrow:after {
  content: " ";
  left: 1px;
  bottom: -10px;
  border-left-width: 0;
  border-right-color: #fff;
}
.popover.bottom > .arrow {
  left: 50%;
  margin-left: -11px;
  border-top-width: 0;
  border-bottom-color: #999999;
  border-bottom-color: rgba(0, 0, 0, 0.25);
  top: -11px;
}
.popover.bottom > .arrow:after {
  content: " ";
  top: 1px;
  margin-left: -10px;
  border-top-width: 0;
  border-bottom-color: #fff;
}
.popover.left > .arrow {
  top: 50%;
  right: -11px;
  margin-top: -11px;
  border-right-width: 0;
  border-left-color: #999999;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.popover.left > .arrow:after {
  content: " ";
  right: 1px;
  border-right-width: 0;
  border-left-color: #fff;
  bottom: -10px;
}
.carousel {
  position: relative;
}
.carousel-inner {
  position: relative;
  overflow: hidden;
  width: 100%;
}
.carousel-inner > .item {
  display: none;
  position: relative;
  -webkit-transition: 0.6s ease-in-out left;
  -o-transition: 0.6s ease-in-out left;
  transition: 0.6s ease-in-out left;
}
.carousel-inner > .item > img,
.carousel-inner > .item > a > img {
  line-height: 1;
}
@media all and (transform-3d), (-webkit-transform-3d) {
  .carousel-inner > .item {
    -webkit-transition: -webkit-transform 0.6s ease-in-out;
    -moz-transition: -moz-transform 0.6s ease-in-out;
    -o-transition: -o-transform 0.6s ease-in-out;
    transition: transform 0.6s ease-in-out;
    -webkit-backface-visibility: hidden;
    -moz-backface-visibility: hidden;
    backface-visibility: hidden;
    -webkit-perspective: 1000px;
    -moz-perspective: 1000px;
    perspective: 1000px;
  }
  .carousel-inner > .item.next,
  .carousel-inner > .item.active.right {
    -webkit-transform: translate3d(100%, 0, 0);
    transform: translate3d(100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.prev,
  .carousel-inner > .item.active.left {
    -webkit-transform: translate3d(-100%, 0, 0);
    transform: translate3d(-100%, 0, 0);
    left: 0;
  }
  .carousel-inner > .item.next.left,
  .carousel-inner > .item.prev.right,
  .carousel-inner > .item.active {
    -webkit-transform: translate3d(0, 0, 0);
    transform: translate3d(0, 0, 0);
    left: 0;
  }
}
.carousel-inner > .active,
.carousel-inner > .next,
.carousel-inner > .prev {
  display: block;
}
.carousel-inner > .active {
  left: 0;
}
.carousel-inner > .next,
.carousel-inner > .prev {
  position: absolute;
  top: 0;
  width: 100%;
}
.carousel-inner > .next {
  left: 100%;
}
.carousel-inner > .prev {
  left: -100%;
}
.carousel-inner > .next.left,
.carousel-inner > .prev.right {
  left: 0;
}
.carousel-inner > .active.left {
  left: -100%;
}
.carousel-inner > .active.right {
  left: 100%;
}
.carousel-control {
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  width: 15%;
  opacity: 0.5;
  filter: alpha(opacity=50);
  font-size: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
  background-color: rgba(0, 0, 0, 0);
}
.carousel-control.left {
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.5) 0%, rgba(0, 0, 0, 0.0001) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#80000000', endColorstr='#00000000', GradientType=1);
}
.carousel-control.right {
  left: auto;
  right: 0;
  background-image: -webkit-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: -o-linear-gradient(left, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0001) 0%, rgba(0, 0, 0, 0.5) 100%);
  background-repeat: repeat-x;
  filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#00000000', endColorstr='#80000000', GradientType=1);
}
.carousel-control:hover,
.carousel-control:focus {
  outline: 0;
  color: #fff;
  text-decoration: none;
  opacity: 0.9;
  filter: alpha(opacity=90);
}
.carousel-control .icon-prev,
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-left,
.carousel-control .glyphicon-chevron-right {
  position: absolute;
  top: 50%;
  margin-top: -10px;
  z-index: 5;
  display: inline-block;
}
.carousel-control .icon-prev,
.carousel-control .glyphicon-chevron-left {
  left: 50%;
  margin-left: -10px;
}
.carousel-control .icon-next,
.carousel-control .glyphicon-chevron-right {
  right: 50%;
  margin-right: -10px;
}
.carousel-control .icon-prev,
.carousel-control .icon-next {
  width: 20px;
  height: 20px;
  line-height: 1;
  font-family: serif;
}
.carousel-control .icon-prev:before {
  content: '\2039';
}
.carousel-control .icon-next:before {
  content: '\203a';
}
.carousel-indicators {
  position: absolute;
  bottom: 10px;
  left: 50%;
  z-index: 15;
  width: 60%;
  margin-left: -30%;
  padding-left: 0;
  list-style: none;
  text-align: center;
}
.carousel-indicators li {
  display: inline-block;
  width: 10px;
  height: 10px;
  margin: 1px;
  text-indent: -999px;
  border: 1px solid #fff;
  border-radius: 10px;
  cursor: pointer;
  background-color: #000 \9;
  background-color: rgba(0, 0, 0, 0);
}
.carousel-indicators .active {
  margin: 0;
  width: 12px;
  height: 12px;
  background-color: #fff;
}
.carousel-caption {
  position: absolute;
  left: 15%;
  right: 15%;
  bottom: 20px;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6);
}
.carousel-caption .btn {
  text-shadow: none;
}
@media screen and (min-width: 768px) {
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-prev,
  .carousel-control .icon-next {
    width: 30px;
    height: 30px;
    margin-top: -10px;
    font-size: 30px;
  }
  .carousel-control .glyphicon-chevron-left,
  .carousel-control .icon-prev {
    margin-left: -10px;
  }
  .carousel-control .glyphicon-chevron-right,
  .carousel-control .icon-next {
    margin-right: -10px;
  }
  .carousel-caption {
    left: 20%;
    right: 20%;
    padding-bottom: 30px;
  }
  .carousel-indicators {
    bottom: 20px;
  }
}
.clearfix:before,
.clearfix:after,
.dl-horizontal dd:before,
.dl-horizontal dd:after,
.container:before,
.container:after,
.container-fluid:before,
.container-fluid:after,
.row:before,
.row:after,
.form-horizontal .form-group:before,
.form-horizontal .form-group:after,
.btn-toolbar:before,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:before,
.btn-group-vertical > .btn-group:after,
.nav:before,
.nav:after,
.navbar:before,
.navbar:after,
.navbar-header:before,
.navbar-header:after,
.navbar-collapse:before,
.navbar-collapse:after,
.pager:before,
.pager:after,
.panel-body:before,
.panel-body:after,
.modal-header:before,
.modal-header:after,
.modal-footer:before,
.modal-footer:after,
.item_buttons:before,
.item_buttons:after {
  content: " ";
  display: table;
}
.clearfix:after,
.dl-horizontal dd:after,
.container:after,
.container-fluid:after,
.row:after,
.form-horizontal .form-group:after,
.btn-toolbar:after,
.btn-group-vertical > .btn-group:after,
.nav:after,
.navbar:after,
.navbar-header:after,
.navbar-collapse:after,
.pager:after,
.panel-body:after,
.modal-header:after,
.modal-footer:after,
.item_buttons:after {
  clear: both;
}
.center-block {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.pull-right {
  float: right !important;
}
.pull-left {
  float: left !important;
}
.hide {
  display: none !important;
}
.show {
  display: block !important;
}
.invisible {
  visibility: hidden;
}
.text-hide {
  font: 0/0 a;
  color: transparent;
  text-shadow: none;
  background-color: transparent;
  border: 0;
}
.hidden {
  display: none !important;
}
.affix {
  position: fixed;
}
@-ms-viewport {
  width: device-width;
}
.visible-xs,
.visible-sm,
.visible-md,
.visible-lg {
  display: none !important;
}
.visible-xs-block,
.visible-xs-inline,
.visible-xs-inline-block,
.visible-sm-block,
.visible-sm-inline,
.visible-sm-inline-block,
.visible-md-block,
.visible-md-inline,
.visible-md-inline-block,
.visible-lg-block,
.visible-lg-inline,
.visible-lg-inline-block {
  display: none !important;
}
@media (max-width: 767px) {
  .visible-xs {
    display: block !important;
  }
  table.visible-xs {
    display: table !important;
  }
  tr.visible-xs {
    display: table-row !important;
  }
  th.visible-xs,
  td.visible-xs {
    display: table-cell !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-block {
    display: block !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline {
    display: inline !important;
  }
}
@media (max-width: 767px) {
  .visible-xs-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm {
    display: block !important;
  }
  table.visible-sm {
    display: table !important;
  }
  tr.visible-sm {
    display: table-row !important;
  }
  th.visible-sm,
  td.visible-sm {
    display: table-cell !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-block {
    display: block !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline {
    display: inline !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .visible-sm-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md {
    display: block !important;
  }
  table.visible-md {
    display: table !important;
  }
  tr.visible-md {
    display: table-row !important;
  }
  th.visible-md,
  td.visible-md {
    display: table-cell !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-block {
    display: block !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline {
    display: inline !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .visible-md-inline-block {
    display: inline-block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg {
    display: block !important;
  }
  table.visible-lg {
    display: table !important;
  }
  tr.visible-lg {
    display: table-row !important;
  }
  th.visible-lg,
  td.visible-lg {
    display: table-cell !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-block {
    display: block !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline {
    display: inline !important;
  }
}
@media (min-width: 1200px) {
  .visible-lg-inline-block {
    display: inline-block !important;
  }
}
@media (max-width: 767px) {
  .hidden-xs {
    display: none !important;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  .hidden-sm {
    display: none !important;
  }
}
@media (min-width: 992px) and (max-width: 1199px) {
  .hidden-md {
    display: none !important;
  }
}
@media (min-width: 1200px) {
  .hidden-lg {
    display: none !important;
  }
}
.visible-print {
  display: none !important;
}
@media print {
  .visible-print {
    display: block !important;
  }
  table.visible-print {
    display: table !important;
  }
  tr.visible-print {
    display: table-row !important;
  }
  th.visible-print,
  td.visible-print {
    display: table-cell !important;
  }
}
.visible-print-block {
  display: none !important;
}
@media print {
  .visible-print-block {
    display: block !important;
  }
}
.visible-print-inline {
  display: none !important;
}
@media print {
  .visible-print-inline {
    display: inline !important;
  }
}
.visible-print-inline-block {
  display: none !important;
}
@media print {
  .visible-print-inline-block {
    display: inline-block !important;
  }
}
@media print {
  .hidden-print {
    display: none !important;
  }
}
/*!
*
* Font Awesome
*
*/
/*!
 *  Font Awesome 4.7.0 by @davegandy - http://fontawesome.io - @fontawesome
 *  License - http://fontawesome.io/license (Font: SIL OFL 1.1, CSS: MIT License)
 */
/* FONT PATH
 * -------------------------- */
@font-face {
  font-family: 'FontAwesome';
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?v=4.7.0');
  src: url('../components/font-awesome/fonts/fontawesome-webfont.eot?#iefix&v=4.7.0') format('embedded-opentype'), url('../components/font-awesome/fonts/fontawesome-webfont.woff2?v=4.7.0') format('woff2'), url('../components/font-awesome/fonts/fontawesome-webfont.woff?v=4.7.0') format('woff'), url('../components/font-awesome/fonts/fontawesome-webfont.ttf?v=4.7.0') format('truetype'), url('../components/font-awesome/fonts/fontawesome-webfont.svg?v=4.7.0#fontawesomeregular') format('svg');
  font-weight: normal;
  font-style: normal;
}
.fa {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
/* makes the font 33% larger relative to the icon container */
.fa-lg {
  font-size: 1.33333333em;
  line-height: 0.75em;
  vertical-align: -15%;
}
.fa-2x {
  font-size: 2em;
}
.fa-3x {
  font-size: 3em;
}
.fa-4x {
  font-size: 4em;
}
.fa-5x {
  font-size: 5em;
}
.fa-fw {
  width: 1.28571429em;
  text-align: center;
}
.fa-ul {
  padding-left: 0;
  margin-left: 2.14285714em;
  list-style-type: none;
}
.fa-ul > li {
  position: relative;
}
.fa-li {
  position: absolute;
  left: -2.14285714em;
  width: 2.14285714em;
  top: 0.14285714em;
  text-align: center;
}
.fa-li.fa-lg {
  left: -1.85714286em;
}
.fa-border {
  padding: .2em .25em .15em;
  border: solid 0.08em #eee;
  border-radius: .1em;
}
.fa-pull-left {
  float: left;
}
.fa-pull-right {
  float: right;
}
.fa.fa-pull-left {
  margin-right: .3em;
}
.fa.fa-pull-right {
  margin-left: .3em;
}
/* Deprecated as of 4.4.0 */
.pull-right {
  float: right;
}
.pull-left {
  float: left;
}
.fa.pull-left {
  margin-right: .3em;
}
.fa.pull-right {
  margin-left: .3em;
}
.fa-spin {
  -webkit-animation: fa-spin 2s infinite linear;
  animation: fa-spin 2s infinite linear;
}
.fa-pulse {
  -webkit-animation: fa-spin 1s infinite steps(8);
  animation: fa-spin 1s infinite steps(8);
}
@-webkit-keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
@keyframes fa-spin {
  0% {
    -webkit-transform: rotate(0deg);
    transform: rotate(0deg);
  }
  100% {
    -webkit-transform: rotate(359deg);
    transform: rotate(359deg);
  }
}
.fa-rotate-90 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}
.fa-rotate-180 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";
  -webkit-transform: rotate(180deg);
  -ms-transform: rotate(180deg);
  transform: rotate(180deg);
}
.fa-rotate-270 {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";
  -webkit-transform: rotate(270deg);
  -ms-transform: rotate(270deg);
  transform: rotate(270deg);
}
.fa-flip-horizontal {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";
  -webkit-transform: scale(-1, 1);
  -ms-transform: scale(-1, 1);
  transform: scale(-1, 1);
}
.fa-flip-vertical {
  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";
  -webkit-transform: scale(1, -1);
  -ms-transform: scale(1, -1);
  transform: scale(1, -1);
}
:root .fa-rotate-90,
:root .fa-rotate-180,
:root .fa-rotate-270,
:root .fa-flip-horizontal,
:root .fa-flip-vertical {
  filter: none;
}
.fa-stack {
  position: relative;
  display: inline-block;
  width: 2em;
  height: 2em;
  line-height: 2em;
  vertical-align: middle;
}
.fa-stack-1x,
.fa-stack-2x {
  position: absolute;
  left: 0;
  width: 100%;
  text-align: center;
}
.fa-stack-1x {
  line-height: inherit;
}
.fa-stack-2x {
  font-size: 2em;
}
.fa-inverse {
  color: #fff;
}
/* Font Awesome uses the Unicode Private Use Area (PUA) to ensure screen
   readers do not read off random characters that represent icons */
.fa-glass:before {
  content: "\f000";
}
.fa-music:before {
  content: "\f001";
}
.fa-search:before {
  content: "\f002";
}
.fa-envelope-o:before {
  content: "\f003";
}
.fa-heart:before {
  content: "\f004";
}
.fa-star:before {
  content: "\f005";
}
.fa-star-o:before {
  content: "\f006";
}
.fa-user:before {
  content: "\f007";
}
.fa-film:before {
  content: "\f008";
}
.fa-th-large:before {
  content: "\f009";
}
.fa-th:before {
  content: "\f00a";
}
.fa-th-list:before {
  content: "\f00b";
}
.fa-check:before {
  content: "\f00c";
}
.fa-remove:before,
.fa-close:before,
.fa-times:before {
  content: "\f00d";
}
.fa-search-plus:before {
  content: "\f00e";
}
.fa-search-minus:before {
  content: "\f010";
}
.fa-power-off:before {
  content: "\f011";
}
.fa-signal:before {
  content: "\f012";
}
.fa-gear:before,
.fa-cog:before {
  content: "\f013";
}
.fa-trash-o:before {
  content: "\f014";
}
.fa-home:before {
  content: "\f015";
}
.fa-file-o:before {
  content: "\f016";
}
.fa-clock-o:before {
  content: "\f017";
}
.fa-road:before {
  content: "\f018";
}
.fa-download:before {
  content: "\f019";
}
.fa-arrow-circle-o-down:before {
  content: "\f01a";
}
.fa-arrow-circle-o-up:before {
  content: "\f01b";
}
.fa-inbox:before {
  content: "\f01c";
}
.fa-play-circle-o:before {
  content: "\f01d";
}
.fa-rotate-right:before,
.fa-repeat:before {
  content: "\f01e";
}
.fa-refresh:before {
  content: "\f021";
}
.fa-list-alt:before {
  content: "\f022";
}
.fa-lock:before {
  content: "\f023";
}
.fa-flag:before {
  content: "\f024";
}
.fa-headphones:before {
  content: "\f025";
}
.fa-volume-off:before {
  content: "\f026";
}
.fa-volume-down:before {
  content: "\f027";
}
.fa-volume-up:before {
  content: "\f028";
}
.fa-qrcode:before {
  content: "\f029";
}
.fa-barcode:before {
  content: "\f02a";
}
.fa-tag:before {
  content: "\f02b";
}
.fa-tags:before {
  content: "\f02c";
}
.fa-book:before {
  content: "\f02d";
}
.fa-bookmark:before {
  content: "\f02e";
}
.fa-print:before {
  content: "\f02f";
}
.fa-camera:before {
  content: "\f030";
}
.fa-font:before {
  content: "\f031";
}
.fa-bold:before {
  content: "\f032";
}
.fa-italic:before {
  content: "\f033";
}
.fa-text-height:before {
  content: "\f034";
}
.fa-text-width:before {
  content: "\f035";
}
.fa-align-left:before {
  content: "\f036";
}
.fa-align-center:before {
  content: "\f037";
}
.fa-align-right:before {
  content: "\f038";
}
.fa-align-justify:before {
  content: "\f039";
}
.fa-list:before {
  content: "\f03a";
}
.fa-dedent:before,
.fa-outdent:before {
  content: "\f03b";
}
.fa-indent:before {
  content: "\f03c";
}
.fa-video-camera:before {
  content: "\f03d";
}
.fa-photo:before,
.fa-image:before,
.fa-picture-o:before {
  content: "\f03e";
}
.fa-pencil:before {
  content: "\f040";
}
.fa-map-marker:before {
  content: "\f041";
}
.fa-adjust:before {
  content: "\f042";
}
.fa-tint:before {
  content: "\f043";
}
.fa-edit:before,
.fa-pencil-square-o:before {
  content: "\f044";
}
.fa-share-square-o:before {
  content: "\f045";
}
.fa-check-square-o:before {
  content: "\f046";
}
.fa-arrows:before {
  content: "\f047";
}
.fa-step-backward:before {
  content: "\f048";
}
.fa-fast-backward:before {
  content: "\f049";
}
.fa-backward:before {
  content: "\f04a";
}
.fa-play:before {
  content: "\f04b";
}
.fa-pause:before {
  content: "\f04c";
}
.fa-stop:before {
  content: "\f04d";
}
.fa-forward:before {
  content: "\f04e";
}
.fa-fast-forward:before {
  content: "\f050";
}
.fa-step-forward:before {
  content: "\f051";
}
.fa-eject:before {
  content: "\f052";
}
.fa-chevron-left:before {
  content: "\f053";
}
.fa-chevron-right:before {
  content: "\f054";
}
.fa-plus-circle:before {
  content: "\f055";
}
.fa-minus-circle:before {
  content: "\f056";
}
.fa-times-circle:before {
  content: "\f057";
}
.fa-check-circle:before {
  content: "\f058";
}
.fa-question-circle:before {
  content: "\f059";
}
.fa-info-circle:before {
  content: "\f05a";
}
.fa-crosshairs:before {
  content: "\f05b";
}
.fa-times-circle-o:before {
  content: "\f05c";
}
.fa-check-circle-o:before {
  content: "\f05d";
}
.fa-ban:before {
  content: "\f05e";
}
.fa-arrow-left:before {
  content: "\f060";
}
.fa-arrow-right:before {
  content: "\f061";
}
.fa-arrow-up:before {
  content: "\f062";
}
.fa-arrow-down:before {
  content: "\f063";
}
.fa-mail-forward:before,
.fa-share:before {
  content: "\f064";
}
.fa-expand:before {
  content: "\f065";
}
.fa-compress:before {
  content: "\f066";
}
.fa-plus:before {
  content: "\f067";
}
.fa-minus:before {
  content: "\f068";
}
.fa-asterisk:before {
  content: "\f069";
}
.fa-exclamation-circle:before {
  content: "\f06a";
}
.fa-gift:before {
  content: "\f06b";
}
.fa-leaf:before {
  content: "\f06c";
}
.fa-fire:before {
  content: "\f06d";
}
.fa-eye:before {
  content: "\f06e";
}
.fa-eye-slash:before {
  content: "\f070";
}
.fa-warning:before,
.fa-exclamation-triangle:before {
  content: "\f071";
}
.fa-plane:before {
  content: "\f072";
}
.fa-calendar:before {
  content: "\f073";
}
.fa-random:before {
  content: "\f074";
}
.fa-comment:before {
  content: "\f075";
}
.fa-magnet:before {
  content: "\f076";
}
.fa-chevron-up:before {
  content: "\f077";
}
.fa-chevron-down:before {
  content: "\f078";
}
.fa-retweet:before {
  content: "\f079";
}
.fa-shopping-cart:before {
  content: "\f07a";
}
.fa-folder:before {
  content: "\f07b";
}
.fa-folder-open:before {
  content: "\f07c";
}
.fa-arrows-v:before {
  content: "\f07d";
}
.fa-arrows-h:before {
  content: "\f07e";
}
.fa-bar-chart-o:before,
.fa-bar-chart:before {
  content: "\f080";
}
.fa-twitter-square:before {
  content: "\f081";
}
.fa-facebook-square:before {
  content: "\f082";
}
.fa-camera-retro:before {
  content: "\f083";
}
.fa-key:before {
  content: "\f084";
}
.fa-gears:before,
.fa-cogs:before {
  content: "\f085";
}
.fa-comments:before {
  content: "\f086";
}
.fa-thumbs-o-up:before {
  content: "\f087";
}
.fa-thumbs-o-down:before {
  content: "\f088";
}
.fa-star-half:before {
  content: "\f089";
}
.fa-heart-o:before {
  content: "\f08a";
}
.fa-sign-out:before {
  content: "\f08b";
}
.fa-linkedin-square:before {
  content: "\f08c";
}
.fa-thumb-tack:before {
  content: "\f08d";
}
.fa-external-link:before {
  content: "\f08e";
}
.fa-sign-in:before {
  content: "\f090";
}
.fa-trophy:before {
  content: "\f091";
}
.fa-github-square:before {
  content: "\f092";
}
.fa-upload:before {
  content: "\f093";
}
.fa-lemon-o:before {
  content: "\f094";
}
.fa-phone:before {
  content: "\f095";
}
.fa-square-o:before {
  content: "\f096";
}
.fa-bookmark-o:before {
  content: "\f097";
}
.fa-phone-square:before {
  content: "\f098";
}
.fa-twitter:before {
  content: "\f099";
}
.fa-facebook-f:before,
.fa-facebook:before {
  content: "\f09a";
}
.fa-github:before {
  content: "\f09b";
}
.fa-unlock:before {
  content: "\f09c";
}
.fa-credit-card:before {
  content: "\f09d";
}
.fa-feed:before,
.fa-rss:before {
  content: "\f09e";
}
.fa-hdd-o:before {
  content: "\f0a0";
}
.fa-bullhorn:before {
  content: "\f0a1";
}
.fa-bell:before {
  content: "\f0f3";
}
.fa-certificate:before {
  content: "\f0a3";
}
.fa-hand-o-right:before {
  content: "\f0a4";
}
.fa-hand-o-left:before {
  content: "\f0a5";
}
.fa-hand-o-up:before {
  content: "\f0a6";
}
.fa-hand-o-down:before {
  content: "\f0a7";
}
.fa-arrow-circle-left:before {
  content: "\f0a8";
}
.fa-arrow-circle-right:before {
  content: "\f0a9";
}
.fa-arrow-circle-up:before {
  content: "\f0aa";
}
.fa-arrow-circle-down:before {
  content: "\f0ab";
}
.fa-globe:before {
  content: "\f0ac";
}
.fa-wrench:before {
  content: "\f0ad";
}
.fa-tasks:before {
  content: "\f0ae";
}
.fa-filter:before {
  content: "\f0b0";
}
.fa-briefcase:before {
  content: "\f0b1";
}
.fa-arrows-alt:before {
  content: "\f0b2";
}
.fa-group:before,
.fa-users:before {
  content: "\f0c0";
}
.fa-chain:before,
.fa-link:before {
  content: "\f0c1";
}
.fa-cloud:before {
  content: "\f0c2";
}
.fa-flask:before {
  content: "\f0c3";
}
.fa-cut:before,
.fa-scissors:before {
  content: "\f0c4";
}
.fa-copy:before,
.fa-files-o:before {
  content: "\f0c5";
}
.fa-paperclip:before {
  content: "\f0c6";
}
.fa-save:before,
.fa-floppy-o:before {
  content: "\f0c7";
}
.fa-square:before {
  content: "\f0c8";
}
.fa-navicon:before,
.fa-reorder:before,
.fa-bars:before {
  content: "\f0c9";
}
.fa-list-ul:before {
  content: "\f0ca";
}
.fa-list-ol:before {
  content: "\f0cb";
}
.fa-strikethrough:before {
  content: "\f0cc";
}
.fa-underline:before {
  content: "\f0cd";
}
.fa-table:before {
  content: "\f0ce";
}
.fa-magic:before {
  content: "\f0d0";
}
.fa-truck:before {
  content: "\f0d1";
}
.fa-pinterest:before {
  content: "\f0d2";
}
.fa-pinterest-square:before {
  content: "\f0d3";
}
.fa-google-plus-square:before {
  content: "\f0d4";
}
.fa-google-plus:before {
  content: "\f0d5";
}
.fa-money:before {
  content: "\f0d6";
}
.fa-caret-down:before {
  content: "\f0d7";
}
.fa-caret-up:before {
  content: "\f0d8";
}
.fa-caret-left:before {
  content: "\f0d9";
}
.fa-caret-right:before {
  content: "\f0da";
}
.fa-columns:before {
  content: "\f0db";
}
.fa-unsorted:before,
.fa-sort:before {
  content: "\f0dc";
}
.fa-sort-down:before,
.fa-sort-desc:before {
  content: "\f0dd";
}
.fa-sort-up:before,
.fa-sort-asc:before {
  content: "\f0de";
}
.fa-envelope:before {
  content: "\f0e0";
}
.fa-linkedin:before {
  content: "\f0e1";
}
.fa-rotate-left:before,
.fa-undo:before {
  content: "\f0e2";
}
.fa-legal:before,
.fa-gavel:before {
  content: "\f0e3";
}
.fa-dashboard:before,
.fa-tachometer:before {
  content: "\f0e4";
}
.fa-comment-o:before {
  content: "\f0e5";
}
.fa-comments-o:before {
  content: "\f0e6";
}
.fa-flash:before,
.fa-bolt:before {
  content: "\f0e7";
}
.fa-sitemap:before {
  content: "\f0e8";
}
.fa-umbrella:before {
  content: "\f0e9";
}
.fa-paste:before,
.fa-clipboard:before {
  content: "\f0ea";
}
.fa-lightbulb-o:before {
  content: "\f0eb";
}
.fa-exchange:before {
  content: "\f0ec";
}
.fa-cloud-download:before {
  content: "\f0ed";
}
.fa-cloud-upload:before {
  content: "\f0ee";
}
.fa-user-md:before {
  content: "\f0f0";
}
.fa-stethoscope:before {
  content: "\f0f1";
}
.fa-suitcase:before {
  content: "\f0f2";
}
.fa-bell-o:before {
  content: "\f0a2";
}
.fa-coffee:before {
  content: "\f0f4";
}
.fa-cutlery:before {
  content: "\f0f5";
}
.fa-file-text-o:before {
  content: "\f0f6";
}
.fa-building-o:before {
  content: "\f0f7";
}
.fa-hospital-o:before {
  content: "\f0f8";
}
.fa-ambulance:before {
  content: "\f0f9";
}
.fa-medkit:before {
  content: "\f0fa";
}
.fa-fighter-jet:before {
  content: "\f0fb";
}
.fa-beer:before {
  content: "\f0fc";
}
.fa-h-square:before {
  content: "\f0fd";
}
.fa-plus-square:before {
  content: "\f0fe";
}
.fa-angle-double-left:before {
  content: "\f100";
}
.fa-angle-double-right:before {
  content: "\f101";
}
.fa-angle-double-up:before {
  content: "\f102";
}
.fa-angle-double-down:before {
  content: "\f103";
}
.fa-angle-left:before {
  content: "\f104";
}
.fa-angle-right:before {
  content: "\f105";
}
.fa-angle-up:before {
  content: "\f106";
}
.fa-angle-down:before {
  content: "\f107";
}
.fa-desktop:before {
  content: "\f108";
}
.fa-laptop:before {
  content: "\f109";
}
.fa-tablet:before {
  content: "\f10a";
}
.fa-mobile-phone:before,
.fa-mobile:before {
  content: "\f10b";
}
.fa-circle-o:before {
  content: "\f10c";
}
.fa-quote-left:before {
  content: "\f10d";
}
.fa-quote-right:before {
  content: "\f10e";
}
.fa-spinner:before {
  content: "\f110";
}
.fa-circle:before {
  content: "\f111";
}
.fa-mail-reply:before,
.fa-reply:before {
  content: "\f112";
}
.fa-github-alt:before {
  content: "\f113";
}
.fa-folder-o:before {
  content: "\f114";
}
.fa-folder-open-o:before {
  content: "\f115";
}
.fa-smile-o:before {
  content: "\f118";
}
.fa-frown-o:before {
  content: "\f119";
}
.fa-meh-o:before {
  content: "\f11a";
}
.fa-gamepad:before {
  content: "\f11b";
}
.fa-keyboard-o:before {
  content: "\f11c";
}
.fa-flag-o:before {
  content: "\f11d";
}
.fa-flag-checkered:before {
  content: "\f11e";
}
.fa-terminal:before {
  content: "\f120";
}
.fa-code:before {
  content: "\f121";
}
.fa-mail-reply-all:before,
.fa-reply-all:before {
  content: "\f122";
}
.fa-star-half-empty:before,
.fa-star-half-full:before,
.fa-star-half-o:before {
  content: "\f123";
}
.fa-location-arrow:before {
  content: "\f124";
}
.fa-crop:before {
  content: "\f125";
}
.fa-code-fork:before {
  content: "\f126";
}
.fa-unlink:before,
.fa-chain-broken:before {
  content: "\f127";
}
.fa-question:before {
  content: "\f128";
}
.fa-info:before {
  content: "\f129";
}
.fa-exclamation:before {
  content: "\f12a";
}
.fa-superscript:before {
  content: "\f12b";
}
.fa-subscript:before {
  content: "\f12c";
}
.fa-eraser:before {
  content: "\f12d";
}
.fa-puzzle-piece:before {
  content: "\f12e";
}
.fa-microphone:before {
  content: "\f130";
}
.fa-microphone-slash:before {
  content: "\f131";
}
.fa-shield:before {
  content: "\f132";
}
.fa-calendar-o:before {
  content: "\f133";
}
.fa-fire-extinguisher:before {
  content: "\f134";
}
.fa-rocket:before {
  content: "\f135";
}
.fa-maxcdn:before {
  content: "\f136";
}
.fa-chevron-circle-left:before {
  content: "\f137";
}
.fa-chevron-circle-right:before {
  content: "\f138";
}
.fa-chevron-circle-up:before {
  content: "\f139";
}
.fa-chevron-circle-down:before {
  content: "\f13a";
}
.fa-html5:before {
  content: "\f13b";
}
.fa-css3:before {
  content: "\f13c";
}
.fa-anchor:before {
  content: "\f13d";
}
.fa-unlock-alt:before {
  content: "\f13e";
}
.fa-bullseye:before {
  content: "\f140";
}
.fa-ellipsis-h:before {
  content: "\f141";
}
.fa-ellipsis-v:before {
  content: "\f142";
}
.fa-rss-square:before {
  content: "\f143";
}
.fa-play-circle:before {
  content: "\f144";
}
.fa-ticket:before {
  content: "\f145";
}
.fa-minus-square:before {
  content: "\f146";
}
.fa-minus-square-o:before {
  content: "\f147";
}
.fa-level-up:before {
  content: "\f148";
}
.fa-level-down:before {
  content: "\f149";
}
.fa-check-square:before {
  content: "\f14a";
}
.fa-pencil-square:before {
  content: "\f14b";
}
.fa-external-link-square:before {
  content: "\f14c";
}
.fa-share-square:before {
  content: "\f14d";
}
.fa-compass:before {
  content: "\f14e";
}
.fa-toggle-down:before,
.fa-caret-square-o-down:before {
  content: "\f150";
}
.fa-toggle-up:before,
.fa-caret-square-o-up:before {
  content: "\f151";
}
.fa-toggle-right:before,
.fa-caret-square-o-right:before {
  content: "\f152";
}
.fa-euro:before,
.fa-eur:before {
  content: "\f153";
}
.fa-gbp:before {
  content: "\f154";
}
.fa-dollar:before,
.fa-usd:before {
  content: "\f155";
}
.fa-rupee:before,
.fa-inr:before {
  content: "\f156";
}
.fa-cny:before,
.fa-rmb:before,
.fa-yen:before,
.fa-jpy:before {
  content: "\f157";
}
.fa-ruble:before,
.fa-rouble:before,
.fa-rub:before {
  content: "\f158";
}
.fa-won:before,
.fa-krw:before {
  content: "\f159";
}
.fa-bitcoin:before,
.fa-btc:before {
  content: "\f15a";
}
.fa-file:before {
  content: "\f15b";
}
.fa-file-text:before {
  content: "\f15c";
}
.fa-sort-alpha-asc:before {
  content: "\f15d";
}
.fa-sort-alpha-desc:before {
  content: "\f15e";
}
.fa-sort-amount-asc:before {
  content: "\f160";
}
.fa-sort-amount-desc:before {
  content: "\f161";
}
.fa-sort-numeric-asc:before {
  content: "\f162";
}
.fa-sort-numeric-desc:before {
  content: "\f163";
}
.fa-thumbs-up:before {
  content: "\f164";
}
.fa-thumbs-down:before {
  content: "\f165";
}
.fa-youtube-square:before {
  content: "\f166";
}
.fa-youtube:before {
  content: "\f167";
}
.fa-xing:before {
  content: "\f168";
}
.fa-xing-square:before {
  content: "\f169";
}
.fa-youtube-play:before {
  content: "\f16a";
}
.fa-dropbox:before {
  content: "\f16b";
}
.fa-stack-overflow:before {
  content: "\f16c";
}
.fa-instagram:before {
  content: "\f16d";
}
.fa-flickr:before {
  content: "\f16e";
}
.fa-adn:before {
  content: "\f170";
}
.fa-bitbucket:before {
  content: "\f171";
}
.fa-bitbucket-square:before {
  content: "\f172";
}
.fa-tumblr:before {
  content: "\f173";
}
.fa-tumblr-square:before {
  content: "\f174";
}
.fa-long-arrow-down:before {
  content: "\f175";
}
.fa-long-arrow-up:before {
  content: "\f176";
}
.fa-long-arrow-left:before {
  content: "\f177";
}
.fa-long-arrow-right:before {
  content: "\f178";
}
.fa-apple:before {
  content: "\f179";
}
.fa-windows:before {
  content: "\f17a";
}
.fa-android:before {
  content: "\f17b";
}
.fa-linux:before {
  content: "\f17c";
}
.fa-dribbble:before {
  content: "\f17d";
}
.fa-skype:before {
  content: "\f17e";
}
.fa-foursquare:before {
  content: "\f180";
}
.fa-trello:before {
  content: "\f181";
}
.fa-female:before {
  content: "\f182";
}
.fa-male:before {
  content: "\f183";
}
.fa-gittip:before,
.fa-gratipay:before {
  content: "\f184";
}
.fa-sun-o:before {
  content: "\f185";
}
.fa-moon-o:before {
  content: "\f186";
}
.fa-archive:before {
  content: "\f187";
}
.fa-bug:before {
  content: "\f188";
}
.fa-vk:before {
  content: "\f189";
}
.fa-weibo:before {
  content: "\f18a";
}
.fa-renren:before {
  content: "\f18b";
}
.fa-pagelines:before {
  content: "\f18c";
}
.fa-stack-exchange:before {
  content: "\f18d";
}
.fa-arrow-circle-o-right:before {
  content: "\f18e";
}
.fa-arrow-circle-o-left:before {
  content: "\f190";
}
.fa-toggle-left:before,
.fa-caret-square-o-left:before {
  content: "\f191";
}
.fa-dot-circle-o:before {
  content: "\f192";
}
.fa-wheelchair:before {
  content: "\f193";
}
.fa-vimeo-square:before {
  content: "\f194";
}
.fa-turkish-lira:before,
.fa-try:before {
  content: "\f195";
}
.fa-plus-square-o:before {
  content: "\f196";
}
.fa-space-shuttle:before {
  content: "\f197";
}
.fa-slack:before {
  content: "\f198";
}
.fa-envelope-square:before {
  content: "\f199";
}
.fa-wordpress:before {
  content: "\f19a";
}
.fa-openid:before {
  content: "\f19b";
}
.fa-institution:before,
.fa-bank:before,
.fa-university:before {
  content: "\f19c";
}
.fa-mortar-board:before,
.fa-graduation-cap:before {
  content: "\f19d";
}
.fa-yahoo:before {
  content: "\f19e";
}
.fa-google:before {
  content: "\f1a0";
}
.fa-reddit:before {
  content: "\f1a1";
}
.fa-reddit-square:before {
  content: "\f1a2";
}
.fa-stumbleupon-circle:before {
  content: "\f1a3";
}
.fa-stumbleupon:before {
  content: "\f1a4";
}
.fa-delicious:before {
  content: "\f1a5";
}
.fa-digg:before {
  content: "\f1a6";
}
.fa-pied-piper-pp:before {
  content: "\f1a7";
}
.fa-pied-piper-alt:before {
  content: "\f1a8";
}
.fa-drupal:before {
  content: "\f1a9";
}
.fa-joomla:before {
  content: "\f1aa";
}
.fa-language:before {
  content: "\f1ab";
}
.fa-fax:before {
  content: "\f1ac";
}
.fa-building:before {
  content: "\f1ad";
}
.fa-child:before {
  content: "\f1ae";
}
.fa-paw:before {
  content: "\f1b0";
}
.fa-spoon:before {
  content: "\f1b1";
}
.fa-cube:before {
  content: "\f1b2";
}
.fa-cubes:before {
  content: "\f1b3";
}
.fa-behance:before {
  content: "\f1b4";
}
.fa-behance-square:before {
  content: "\f1b5";
}
.fa-steam:before {
  content: "\f1b6";
}
.fa-steam-square:before {
  content: "\f1b7";
}
.fa-recycle:before {
  content: "\f1b8";
}
.fa-automobile:before,
.fa-car:before {
  content: "\f1b9";
}
.fa-cab:before,
.fa-taxi:before {
  content: "\f1ba";
}
.fa-tree:before {
  content: "\f1bb";
}
.fa-spotify:before {
  content: "\f1bc";
}
.fa-deviantart:before {
  content: "\f1bd";
}
.fa-soundcloud:before {
  content: "\f1be";
}
.fa-database:before {
  content: "\f1c0";
}
.fa-file-pdf-o:before {
  content: "\f1c1";
}
.fa-file-word-o:before {
  content: "\f1c2";
}
.fa-file-excel-o:before {
  content: "\f1c3";
}
.fa-file-powerpoint-o:before {
  content: "\f1c4";
}
.fa-file-photo-o:before,
.fa-file-picture-o:before,
.fa-file-image-o:before {
  content: "\f1c5";
}
.fa-file-zip-o:before,
.fa-file-archive-o:before {
  content: "\f1c6";
}
.fa-file-sound-o:before,
.fa-file-audio-o:before {
  content: "\f1c7";
}
.fa-file-movie-o:before,
.fa-file-video-o:before {
  content: "\f1c8";
}
.fa-file-code-o:before {
  content: "\f1c9";
}
.fa-vine:before {
  content: "\f1ca";
}
.fa-codepen:before {
  content: "\f1cb";
}
.fa-jsfiddle:before {
  content: "\f1cc";
}
.fa-life-bouy:before,
.fa-life-buoy:before,
.fa-life-saver:before,
.fa-support:before,
.fa-life-ring:before {
  content: "\f1cd";
}
.fa-circle-o-notch:before {
  content: "\f1ce";
}
.fa-ra:before,
.fa-resistance:before,
.fa-rebel:before {
  content: "\f1d0";
}
.fa-ge:before,
.fa-empire:before {
  content: "\f1d1";
}
.fa-git-square:before {
  content: "\f1d2";
}
.fa-git:before {
  content: "\f1d3";
}
.fa-y-combinator-square:before,
.fa-yc-square:before,
.fa-hacker-news:before {
  content: "\f1d4";
}
.fa-tencent-weibo:before {
  content: "\f1d5";
}
.fa-qq:before {
  content: "\f1d6";
}
.fa-wechat:before,
.fa-weixin:before {
  content: "\f1d7";
}
.fa-send:before,
.fa-paper-plane:before {
  content: "\f1d8";
}
.fa-send-o:before,
.fa-paper-plane-o:before {
  content: "\f1d9";
}
.fa-history:before {
  content: "\f1da";
}
.fa-circle-thin:before {
  content: "\f1db";
}
.fa-header:before {
  content: "\f1dc";
}
.fa-paragraph:before {
  content: "\f1dd";
}
.fa-sliders:before {
  content: "\f1de";
}
.fa-share-alt:before {
  content: "\f1e0";
}
.fa-share-alt-square:before {
  content: "\f1e1";
}
.fa-bomb:before {
  content: "\f1e2";
}
.fa-soccer-ball-o:before,
.fa-futbol-o:before {
  content: "\f1e3";
}
.fa-tty:before {
  content: "\f1e4";
}
.fa-binoculars:before {
  content: "\f1e5";
}
.fa-plug:before {
  content: "\f1e6";
}
.fa-slideshare:before {
  content: "\f1e7";
}
.fa-twitch:before {
  content: "\f1e8";
}
.fa-yelp:before {
  content: "\f1e9";
}
.fa-newspaper-o:before {
  content: "\f1ea";
}
.fa-wifi:before {
  content: "\f1eb";
}
.fa-calculator:before {
  content: "\f1ec";
}
.fa-paypal:before {
  content: "\f1ed";
}
.fa-google-wallet:before {
  content: "\f1ee";
}
.fa-cc-visa:before {
  content: "\f1f0";
}
.fa-cc-mastercard:before {
  content: "\f1f1";
}
.fa-cc-discover:before {
  content: "\f1f2";
}
.fa-cc-amex:before {
  content: "\f1f3";
}
.fa-cc-paypal:before {
  content: "\f1f4";
}
.fa-cc-stripe:before {
  content: "\f1f5";
}
.fa-bell-slash:before {
  content: "\f1f6";
}
.fa-bell-slash-o:before {
  content: "\f1f7";
}
.fa-trash:before {
  content: "\f1f8";
}
.fa-copyright:before {
  content: "\f1f9";
}
.fa-at:before {
  content: "\f1fa";
}
.fa-eyedropper:before {
  content: "\f1fb";
}
.fa-paint-brush:before {
  content: "\f1fc";
}
.fa-birthday-cake:before {
  content: "\f1fd";
}
.fa-area-chart:before {
  content: "\f1fe";
}
.fa-pie-chart:before {
  content: "\f200";
}
.fa-line-chart:before {
  content: "\f201";
}
.fa-lastfm:before {
  content: "\f202";
}
.fa-lastfm-square:before {
  content: "\f203";
}
.fa-toggle-off:before {
  content: "\f204";
}
.fa-toggle-on:before {
  content: "\f205";
}
.fa-bicycle:before {
  content: "\f206";
}
.fa-bus:before {
  content: "\f207";
}
.fa-ioxhost:before {
  content: "\f208";
}
.fa-angellist:before {
  content: "\f209";
}
.fa-cc:before {
  content: "\f20a";
}
.fa-shekel:before,
.fa-sheqel:before,
.fa-ils:before {
  content: "\f20b";
}
.fa-meanpath:before {
  content: "\f20c";
}
.fa-buysellads:before {
  content: "\f20d";
}
.fa-connectdevelop:before {
  content: "\f20e";
}
.fa-dashcube:before {
  content: "\f210";
}
.fa-forumbee:before {
  content: "\f211";
}
.fa-leanpub:before {
  content: "\f212";
}
.fa-sellsy:before {
  content: "\f213";
}
.fa-shirtsinbulk:before {
  content: "\f214";
}
.fa-simplybuilt:before {
  content: "\f215";
}
.fa-skyatlas:before {
  content: "\f216";
}
.fa-cart-plus:before {
  content: "\f217";
}
.fa-cart-arrow-down:before {
  content: "\f218";
}
.fa-diamond:before {
  content: "\f219";
}
.fa-ship:before {
  content: "\f21a";
}
.fa-user-secret:before {
  content: "\f21b";
}
.fa-motorcycle:before {
  content: "\f21c";
}
.fa-street-view:before {
  content: "\f21d";
}
.fa-heartbeat:before {
  content: "\f21e";
}
.fa-venus:before {
  content: "\f221";
}
.fa-mars:before {
  content: "\f222";
}
.fa-mercury:before {
  content: "\f223";
}
.fa-intersex:before,
.fa-transgender:before {
  content: "\f224";
}
.fa-transgender-alt:before {
  content: "\f225";
}
.fa-venus-double:before {
  content: "\f226";
}
.fa-mars-double:before {
  content: "\f227";
}
.fa-venus-mars:before {
  content: "\f228";
}
.fa-mars-stroke:before {
  content: "\f229";
}
.fa-mars-stroke-v:before {
  content: "\f22a";
}
.fa-mars-stroke-h:before {
  content: "\f22b";
}
.fa-neuter:before {
  content: "\f22c";
}
.fa-genderless:before {
  content: "\f22d";
}
.fa-facebook-official:before {
  content: "\f230";
}
.fa-pinterest-p:before {
  content: "\f231";
}
.fa-whatsapp:before {
  content: "\f232";
}
.fa-server:before {
  content: "\f233";
}
.fa-user-plus:before {
  content: "\f234";
}
.fa-user-times:before {
  content: "\f235";
}
.fa-hotel:before,
.fa-bed:before {
  content: "\f236";
}
.fa-viacoin:before {
  content: "\f237";
}
.fa-train:before {
  content: "\f238";
}
.fa-subway:before {
  content: "\f239";
}
.fa-medium:before {
  content: "\f23a";
}
.fa-yc:before,
.fa-y-combinator:before {
  content: "\f23b";
}
.fa-optin-monster:before {
  content: "\f23c";
}
.fa-opencart:before {
  content: "\f23d";
}
.fa-expeditedssl:before {
  content: "\f23e";
}
.fa-battery-4:before,
.fa-battery:before,
.fa-battery-full:before {
  content: "\f240";
}
.fa-battery-3:before,
.fa-battery-three-quarters:before {
  content: "\f241";
}
.fa-battery-2:before,
.fa-battery-half:before {
  content: "\f242";
}
.fa-battery-1:before,
.fa-battery-quarter:before {
  content: "\f243";
}
.fa-battery-0:before,
.fa-battery-empty:before {
  content: "\f244";
}
.fa-mouse-pointer:before {
  content: "\f245";
}
.fa-i-cursor:before {
  content: "\f246";
}
.fa-object-group:before {
  content: "\f247";
}
.fa-object-ungroup:before {
  content: "\f248";
}
.fa-sticky-note:before {
  content: "\f249";
}
.fa-sticky-note-o:before {
  content: "\f24a";
}
.fa-cc-jcb:before {
  content: "\f24b";
}
.fa-cc-diners-club:before {
  content: "\f24c";
}
.fa-clone:before {
  content: "\f24d";
}
.fa-balance-scale:before {
  content: "\f24e";
}
.fa-hourglass-o:before {
  content: "\f250";
}
.fa-hourglass-1:before,
.fa-hourglass-start:before {
  content: "\f251";
}
.fa-hourglass-2:before,
.fa-hourglass-half:before {
  content: "\f252";
}
.fa-hourglass-3:before,
.fa-hourglass-end:before {
  content: "\f253";
}
.fa-hourglass:before {
  content: "\f254";
}
.fa-hand-grab-o:before,
.fa-hand-rock-o:before {
  content: "\f255";
}
.fa-hand-stop-o:before,
.fa-hand-paper-o:before {
  content: "\f256";
}
.fa-hand-scissors-o:before {
  content: "\f257";
}
.fa-hand-lizard-o:before {
  content: "\f258";
}
.fa-hand-spock-o:before {
  content: "\f259";
}
.fa-hand-pointer-o:before {
  content: "\f25a";
}
.fa-hand-peace-o:before {
  content: "\f25b";
}
.fa-trademark:before {
  content: "\f25c";
}
.fa-registered:before {
  content: "\f25d";
}
.fa-creative-commons:before {
  content: "\f25e";
}
.fa-gg:before {
  content: "\f260";
}
.fa-gg-circle:before {
  content: "\f261";
}
.fa-tripadvisor:before {
  content: "\f262";
}
.fa-odnoklassniki:before {
  content: "\f263";
}
.fa-odnoklassniki-square:before {
  content: "\f264";
}
.fa-get-pocket:before {
  content: "\f265";
}
.fa-wikipedia-w:before {
  content: "\f266";
}
.fa-safari:before {
  content: "\f267";
}
.fa-chrome:before {
  content: "\f268";
}
.fa-firefox:before {
  content: "\f269";
}
.fa-opera:before {
  content: "\f26a";
}
.fa-internet-explorer:before {
  content: "\f26b";
}
.fa-tv:before,
.fa-television:before {
  content: "\f26c";
}
.fa-contao:before {
  content: "\f26d";
}
.fa-500px:before {
  content: "\f26e";
}
.fa-amazon:before {
  content: "\f270";
}
.fa-calendar-plus-o:before {
  content: "\f271";
}
.fa-calendar-minus-o:before {
  content: "\f272";
}
.fa-calendar-times-o:before {
  content: "\f273";
}
.fa-calendar-check-o:before {
  content: "\f274";
}
.fa-industry:before {
  content: "\f275";
}
.fa-map-pin:before {
  content: "\f276";
}
.fa-map-signs:before {
  content: "\f277";
}
.fa-map-o:before {
  content: "\f278";
}
.fa-map:before {
  content: "\f279";
}
.fa-commenting:before {
  content: "\f27a";
}
.fa-commenting-o:before {
  content: "\f27b";
}
.fa-houzz:before {
  content: "\f27c";
}
.fa-vimeo:before {
  content: "\f27d";
}
.fa-black-tie:before {
  content: "\f27e";
}
.fa-fonticons:before {
  content: "\f280";
}
.fa-reddit-alien:before {
  content: "\f281";
}
.fa-edge:before {
  content: "\f282";
}
.fa-credit-card-alt:before {
  content: "\f283";
}
.fa-codiepie:before {
  content: "\f284";
}
.fa-modx:before {
  content: "\f285";
}
.fa-fort-awesome:before {
  content: "\f286";
}
.fa-usb:before {
  content: "\f287";
}
.fa-product-hunt:before {
  content: "\f288";
}
.fa-mixcloud:before {
  content: "\f289";
}
.fa-scribd:before {
  content: "\f28a";
}
.fa-pause-circle:before {
  content: "\f28b";
}
.fa-pause-circle-o:before {
  content: "\f28c";
}
.fa-stop-circle:before {
  content: "\f28d";
}
.fa-stop-circle-o:before {
  content: "\f28e";
}
.fa-shopping-bag:before {
  content: "\f290";
}
.fa-shopping-basket:before {
  content: "\f291";
}
.fa-hashtag:before {
  content: "\f292";
}
.fa-bluetooth:before {
  content: "\f293";
}
.fa-bluetooth-b:before {
  content: "\f294";
}
.fa-percent:before {
  content: "\f295";
}
.fa-gitlab:before {
  content: "\f296";
}
.fa-wpbeginner:before {
  content: "\f297";
}
.fa-wpforms:before {
  content: "\f298";
}
.fa-envira:before {
  content: "\f299";
}
.fa-universal-access:before {
  content: "\f29a";
}
.fa-wheelchair-alt:before {
  content: "\f29b";
}
.fa-question-circle-o:before {
  content: "\f29c";
}
.fa-blind:before {
  content: "\f29d";
}
.fa-audio-description:before {
  content: "\f29e";
}
.fa-volume-control-phone:before {
  content: "\f2a0";
}
.fa-braille:before {
  content: "\f2a1";
}
.fa-assistive-listening-systems:before {
  content: "\f2a2";
}
.fa-asl-interpreting:before,
.fa-american-sign-language-interpreting:before {
  content: "\f2a3";
}
.fa-deafness:before,
.fa-hard-of-hearing:before,
.fa-deaf:before {
  content: "\f2a4";
}
.fa-glide:before {
  content: "\f2a5";
}
.fa-glide-g:before {
  content: "\f2a6";
}
.fa-signing:before,
.fa-sign-language:before {
  content: "\f2a7";
}
.fa-low-vision:before {
  content: "\f2a8";
}
.fa-viadeo:before {
  content: "\f2a9";
}
.fa-viadeo-square:before {
  content: "\f2aa";
}
.fa-snapchat:before {
  content: "\f2ab";
}
.fa-snapchat-ghost:before {
  content: "\f2ac";
}
.fa-snapchat-square:before {
  content: "\f2ad";
}
.fa-pied-piper:before {
  content: "\f2ae";
}
.fa-first-order:before {
  content: "\f2b0";
}
.fa-yoast:before {
  content: "\f2b1";
}
.fa-themeisle:before {
  content: "\f2b2";
}
.fa-google-plus-circle:before,
.fa-google-plus-official:before {
  content: "\f2b3";
}
.fa-fa:before,
.fa-font-awesome:before {
  content: "\f2b4";
}
.fa-handshake-o:before {
  content: "\f2b5";
}
.fa-envelope-open:before {
  content: "\f2b6";
}
.fa-envelope-open-o:before {
  content: "\f2b7";
}
.fa-linode:before {
  content: "\f2b8";
}
.fa-address-book:before {
  content: "\f2b9";
}
.fa-address-book-o:before {
  content: "\f2ba";
}
.fa-vcard:before,
.fa-address-card:before {
  content: "\f2bb";
}
.fa-vcard-o:before,
.fa-address-card-o:before {
  content: "\f2bc";
}
.fa-user-circle:before {
  content: "\f2bd";
}
.fa-user-circle-o:before {
  content: "\f2be";
}
.fa-user-o:before {
  content: "\f2c0";
}
.fa-id-badge:before {
  content: "\f2c1";
}
.fa-drivers-license:before,
.fa-id-card:before {
  content: "\f2c2";
}
.fa-drivers-license-o:before,
.fa-id-card-o:before {
  content: "\f2c3";
}
.fa-quora:before {
  content: "\f2c4";
}
.fa-free-code-camp:before {
  content: "\f2c5";
}
.fa-telegram:before {
  content: "\f2c6";
}
.fa-thermometer-4:before,
.fa-thermometer:before,
.fa-thermometer-full:before {
  content: "\f2c7";
}
.fa-thermometer-3:before,
.fa-thermometer-three-quarters:before {
  content: "\f2c8";
}
.fa-thermometer-2:before,
.fa-thermometer-half:before {
  content: "\f2c9";
}
.fa-thermometer-1:before,
.fa-thermometer-quarter:before {
  content: "\f2ca";
}
.fa-thermometer-0:before,
.fa-thermometer-empty:before {
  content: "\f2cb";
}
.fa-shower:before {
  content: "\f2cc";
}
.fa-bathtub:before,
.fa-s15:before,
.fa-bath:before {
  content: "\f2cd";
}
.fa-podcast:before {
  content: "\f2ce";
}
.fa-window-maximize:before {
  content: "\f2d0";
}
.fa-window-minimize:before {
  content: "\f2d1";
}
.fa-window-restore:before {
  content: "\f2d2";
}
.fa-times-rectangle:before,
.fa-window-close:before {
  content: "\f2d3";
}
.fa-times-rectangle-o:before,
.fa-window-close-o:before {
  content: "\f2d4";
}
.fa-bandcamp:before {
  content: "\f2d5";
}
.fa-grav:before {
  content: "\f2d6";
}
.fa-etsy:before {
  content: "\f2d7";
}
.fa-imdb:before {
  content: "\f2d8";
}
.fa-ravelry:before {
  content: "\f2d9";
}
.fa-eercast:before {
  content: "\f2da";
}
.fa-microchip:before {
  content: "\f2db";
}
.fa-snowflake-o:before {
  content: "\f2dc";
}
.fa-superpowers:before {
  content: "\f2dd";
}
.fa-wpexplorer:before {
  content: "\f2de";
}
.fa-meetup:before {
  content: "\f2e0";
}
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
.sr-only-focusable:active,
.sr-only-focusable:focus {
  position: static;
  width: auto;
  height: auto;
  margin: 0;
  overflow: visible;
  clip: auto;
}
/*!
*
* IPython base
*
*/
.modal.fade .modal-dialog {
  -webkit-transform: translate(0, 0);
  -ms-transform: translate(0, 0);
  -o-transform: translate(0, 0);
  transform: translate(0, 0);
}
code {
  color: #000;
}
pre {
  font-size: inherit;
  line-height: inherit;
}
label {
  font-weight: normal;
}
/* Make the page background atleast 100% the height of the view port */
/* Make the page itself atleast 70% the height of the view port */
.border-box-sizing {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.corner-all {
  border-radius: 2px;
}
.no-padding {
  padding: 0px;
}
/* Flexible box model classes */
/* Taken from Alex Russell http://infrequently.org/2009/08/css-3-progress/ */
/* This file is a compatability layer.  It allows the usage of flexible box 
model layouts accross multiple browsers, including older browsers.  The newest,
universal implementation of the flexible box model is used when available (see
`Modern browsers` comments below).  Browsers that are known to implement this 
new spec completely include:

    Firefox 28.0+
    Chrome 29.0+
    Internet Explorer 11+ 
    Opera 17.0+

Browsers not listed, including Safari, are supported via the styling under the
`Old browsers` comments below.
*/
.hbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
.hbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.vbox {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
.vbox > * {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
}
.hbox.reverse,
.vbox.reverse,
.reverse {
  /* Old browsers */
  -webkit-box-direction: reverse;
  -moz-box-direction: reverse;
  box-direction: reverse;
  /* Modern browsers */
  flex-direction: row-reverse;
}
.hbox.box-flex0,
.vbox.box-flex0,
.box-flex0 {
  /* Old browsers */
  -webkit-box-flex: 0;
  -moz-box-flex: 0;
  box-flex: 0;
  /* Modern browsers */
  flex: none;
  width: auto;
}
.hbox.box-flex1,
.vbox.box-flex1,
.box-flex1 {
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex,
.vbox.box-flex,
.box-flex {
  /* Old browsers */
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
.hbox.box-flex2,
.vbox.box-flex2,
.box-flex2 {
  /* Old browsers */
  -webkit-box-flex: 2;
  -moz-box-flex: 2;
  box-flex: 2;
  /* Modern browsers */
  flex: 2;
}
.box-group1 {
  /*  Deprecated */
  -webkit-box-flex-group: 1;
  -moz-box-flex-group: 1;
  box-flex-group: 1;
}
.box-group2 {
  /* Deprecated */
  -webkit-box-flex-group: 2;
  -moz-box-flex-group: 2;
  box-flex-group: 2;
}
.hbox.start,
.vbox.start,
.start {
  /* Old browsers */
  -webkit-box-pack: start;
  -moz-box-pack: start;
  box-pack: start;
  /* Modern browsers */
  justify-content: flex-start;
}
.hbox.end,
.vbox.end,
.end {
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
}
.hbox.center,
.vbox.center,
.center {
  /* Old browsers */
  -webkit-box-pack: center;
  -moz-box-pack: center;
  box-pack: center;
  /* Modern browsers */
  justify-content: center;
}
.hbox.baseline,
.vbox.baseline,
.baseline {
  /* Old browsers */
  -webkit-box-pack: baseline;
  -moz-box-pack: baseline;
  box-pack: baseline;
  /* Modern browsers */
  justify-content: baseline;
}
.hbox.stretch,
.vbox.stretch,
.stretch {
  /* Old browsers */
  -webkit-box-pack: stretch;
  -moz-box-pack: stretch;
  box-pack: stretch;
  /* Modern browsers */
  justify-content: stretch;
}
.hbox.align-start,
.vbox.align-start,
.align-start {
  /* Old browsers */
  -webkit-box-align: start;
  -moz-box-align: start;
  box-align: start;
  /* Modern browsers */
  align-items: flex-start;
}
.hbox.align-end,
.vbox.align-end,
.align-end {
  /* Old browsers */
  -webkit-box-align: end;
  -moz-box-align: end;
  box-align: end;
  /* Modern browsers */
  align-items: flex-end;
}
.hbox.align-center,
.vbox.align-center,
.align-center {
  /* Old browsers */
  -webkit-box-align: center;
  -moz-box-align: center;
  box-align: center;
  /* Modern browsers */
  align-items: center;
}
.hbox.align-baseline,
.vbox.align-baseline,
.align-baseline {
  /* Old browsers */
  -webkit-box-align: baseline;
  -moz-box-align: baseline;
  box-align: baseline;
  /* Modern browsers */
  align-items: baseline;
}
.hbox.align-stretch,
.vbox.align-stretch,
.align-stretch {
  /* Old browsers */
  -webkit-box-align: stretch;
  -moz-box-align: stretch;
  box-align: stretch;
  /* Modern browsers */
  align-items: stretch;
}
div.error {
  margin: 2em;
  text-align: center;
}
div.error > h1 {
  font-size: 500%;
  line-height: normal;
}
div.error > p {
  font-size: 200%;
  line-height: normal;
}
div.traceback-wrapper {
  text-align: left;
  max-width: 800px;
  margin: auto;
}
div.traceback-wrapper pre.traceback {
  max-height: 600px;
  overflow: auto;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
body {
  background-color: #fff;
  /* This makes sure that the body covers the entire window and needs to
       be in a different element than the display: box in wrapper below */
  position: absolute;
  left: 0px;
  right: 0px;
  top: 0px;
  bottom: 0px;
  overflow: visible;
}
body > #header {
  /* Initially hidden to prevent FLOUC */
  display: none;
  background-color: #fff;
  /* Display over codemirror */
  position: relative;
  z-index: 100;
}
body > #header #header-container {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  padding: 5px;
  padding-bottom: 5px;
  padding-top: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
body > #header .header-bar {
  width: 100%;
  height: 1px;
  background: #e7e7e7;
  margin-bottom: -1px;
}
@media print {
  body > #header {
    display: none !important;
  }
}
#header-spacer {
  width: 100%;
  visibility: hidden;
}
@media print {
  #header-spacer {
    display: none;
  }
}
#ipython_notebook {
  padding-left: 0px;
  padding-top: 1px;
  padding-bottom: 1px;
}
[dir="rtl"] #ipython_notebook {
  margin-right: 10px;
  margin-left: 0;
}
[dir="rtl"] #ipython_notebook.pull-left {
  float: right !important;
  float: right;
}
.flex-spacer {
  flex: 1;
}
#noscript {
  width: auto;
  padding-top: 16px;
  padding-bottom: 16px;
  text-align: center;
  font-size: 22px;
  color: red;
  font-weight: bold;
}
#ipython_notebook img {
  height: 28px;
}
#site {
  width: 100%;
  display: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  overflow: auto;
}
@media print {
  #site {
    height: auto !important;
  }
}
/* Smaller buttons */
.ui-button .ui-button-text {
  padding: 0.2em 0.8em;
  font-size: 77%;
}
input.ui-button {
  padding: 0.3em 0.9em;
}
span#kernel_logo_widget {
  margin: 0 10px;
}
span#login_widget {
  float: right;
}
[dir="rtl"] span#login_widget {
  float: left;
}
span#login_widget > .button,
#logout {
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button:focus,
#logout:focus,
span#login_widget > .button.focus,
#logout.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
span#login_widget > .button:hover,
#logout:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
span#login_widget > .button:active:hover,
#logout:active:hover,
span#login_widget > .button.active:hover,
#logout.active:hover,
.open > .dropdown-togglespan#login_widget > .button:hover,
.open > .dropdown-toggle#logout:hover,
span#login_widget > .button:active:focus,
#logout:active:focus,
span#login_widget > .button.active:focus,
#logout.active:focus,
.open > .dropdown-togglespan#login_widget > .button:focus,
.open > .dropdown-toggle#logout:focus,
span#login_widget > .button:active.focus,
#logout:active.focus,
span#login_widget > .button.active.focus,
#logout.active.focus,
.open > .dropdown-togglespan#login_widget > .button.focus,
.open > .dropdown-toggle#logout.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
span#login_widget > .button:active,
#logout:active,
span#login_widget > .button.active,
#logout.active,
.open > .dropdown-togglespan#login_widget > .button,
.open > .dropdown-toggle#logout {
  background-image: none;
}
span#login_widget > .button.disabled:hover,
#logout.disabled:hover,
span#login_widget > .button[disabled]:hover,
#logout[disabled]:hover,
fieldset[disabled] span#login_widget > .button:hover,
fieldset[disabled] #logout:hover,
span#login_widget > .button.disabled:focus,
#logout.disabled:focus,
span#login_widget > .button[disabled]:focus,
#logout[disabled]:focus,
fieldset[disabled] span#login_widget > .button:focus,
fieldset[disabled] #logout:focus,
span#login_widget > .button.disabled.focus,
#logout.disabled.focus,
span#login_widget > .button[disabled].focus,
#logout[disabled].focus,
fieldset[disabled] span#login_widget > .button.focus,
fieldset[disabled] #logout.focus {
  background-color: #fff;
  border-color: #ccc;
}
span#login_widget > .button .badge,
#logout .badge {
  color: #fff;
  background-color: #333;
}
.nav-header {
  text-transform: none;
}
#header > span {
  margin-top: 10px;
}
.modal_stretch .modal-dialog {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  min-height: 80vh;
}
.modal_stretch .modal-dialog .modal-body {
  max-height: calc(100vh - 200px);
  overflow: auto;
  flex: 1;
}
.modal-header {
  cursor: move;
}
@media (min-width: 768px) {
  .modal .modal-dialog {
    width: 700px;
  }
}
@media (min-width: 768px) {
  select.form-control {
    margin-left: 12px;
    margin-right: 12px;
  }
}
/*!
*
* IPython auth
*
*/
.center-nav {
  display: inline-block;
  margin-bottom: -4px;
}
[dir="rtl"] .center-nav form.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] .center-nav .navbar-text {
  float: right;
}
[dir="rtl"] .navbar-inner {
  text-align: right;
}
[dir="rtl"] div.text-left {
  text-align: right;
}
/*!
*
* IPython tree view
*
*/
/* We need an invisible input field on top of the sentense*/
/* "Drag file onto the list ..." */
.alternate_upload {
  background-color: none;
  display: inline;
}
.alternate_upload.form {
  padding: 0;
  margin: 0;
}
.alternate_upload input.fileinput {
  position: absolute;
  display: block;
  width: 100%;
  height: 100%;
  overflow: hidden;
  cursor: pointer;
  opacity: 0;
  z-index: 2;
}
.alternate_upload .btn-xs > input.fileinput {
  margin: -1px -5px;
}
.alternate_upload .btn-upload {
  position: relative;
  height: 22px;
}
::-webkit-file-upload-button {
  cursor: pointer;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
ul#tabs {
  margin-bottom: 4px;
}
ul#tabs a {
  padding-top: 6px;
  padding-bottom: 4px;
}
[dir="rtl"] ul#tabs.nav-tabs > li {
  float: right;
}
[dir="rtl"] ul#tabs.nav.nav-tabs {
  padding-right: 0;
}
ul.breadcrumb a:focus,
ul.breadcrumb a:hover {
  text-decoration: none;
}
ul.breadcrumb i.icon-home {
  font-size: 16px;
  margin-right: 4px;
}
ul.breadcrumb span {
  color: #5e5e5e;
}
.list_toolbar {
  padding: 4px 0 4px 0;
  vertical-align: middle;
}
.list_toolbar .tree-buttons {
  padding-top: 1px;
}
[dir="rtl"] .list_toolbar .tree-buttons .pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .list_toolbar .col-sm-4,
[dir="rtl"] .list_toolbar .col-sm-8 {
  float: right;
}
.dynamic-buttons {
  padding-top: 3px;
  display: inline-block;
}
.list_toolbar [class*="span"] {
  min-height: 24px;
}
.list_header {
  font-weight: bold;
  background-color: #EEE;
}
.list_placeholder {
  font-weight: bold;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
}
.list_container {
  margin-top: 4px;
  margin-bottom: 20px;
  border: 1px solid #ddd;
  border-radius: 2px;
}
.list_container > div {
  border-bottom: 1px solid #ddd;
}
.list_container > div:hover .list-item {
  background-color: red;
}
.list_container > div:last-child {
  border: none;
}
.list_item:hover .list_item {
  background-color: #ddd;
}
.list_item a {
  text-decoration: none;
}
.list_item:hover {
  background-color: #fafafa;
}
.list_header > div,
.list_item > div {
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
.list_header > div input,
.list_item > div input {
  margin-right: 7px;
  margin-left: 14px;
  vertical-align: text-bottom;
  line-height: 22px;
  position: relative;
  top: -1px;
}
.list_header > div .item_link,
.list_item > div .item_link {
  margin-left: -1px;
  vertical-align: baseline;
  line-height: 22px;
}
[dir="rtl"] .list_item > div input {
  margin-right: 0;
}
.new-file input[type=checkbox] {
  visibility: hidden;
}
.item_name {
  line-height: 22px;
  height: 24px;
}
.item_icon {
  font-size: 14px;
  color: #5e5e5e;
  margin-right: 7px;
  margin-left: 7px;
  line-height: 22px;
  vertical-align: baseline;
}
.item_modified {
  margin-right: 7px;
  margin-left: 7px;
}
[dir="rtl"] .item_modified.pull-right {
  float: left !important;
  float: left;
}
.item_buttons {
  line-height: 1em;
  margin-left: -5px;
}
.item_buttons .btn,
.item_buttons .btn-group,
.item_buttons .input-group {
  float: left;
}
.item_buttons > .btn,
.item_buttons > .btn-group,
.item_buttons > .input-group {
  margin-left: 5px;
}
.item_buttons .btn {
  min-width: 13ex;
}
.item_buttons .running-indicator {
  padding-top: 4px;
  color: #5cb85c;
}
.item_buttons .kernel-name {
  padding-top: 4px;
  color: #5bc0de;
  margin-right: 7px;
  float: left;
}
[dir="rtl"] .item_buttons.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .item_buttons .kernel-name {
  margin-left: 7px;
  float: right;
}
.toolbar_info {
  height: 24px;
  line-height: 24px;
}
.list_item input:not([type=checkbox]) {
  padding-top: 3px;
  padding-bottom: 3px;
  height: 22px;
  line-height: 14px;
  margin: 0px;
}
.highlight_text {
  color: blue;
}
#project_name {
  display: inline-block;
  padding-left: 7px;
  margin-left: -2px;
}
#project_name > .breadcrumb {
  padding: 0px;
  margin-bottom: 0px;
  background-color: transparent;
  font-weight: bold;
}
.sort_button {
  display: inline-block;
  padding-left: 7px;
}
[dir="rtl"] .sort_button.pull-right {
  float: left !important;
  float: left;
}
#tree-selector {
  padding-right: 0px;
}
#button-select-all {
  min-width: 50px;
}
[dir="rtl"] #button-select-all.btn {
  float: right ;
}
#select-all {
  margin-left: 7px;
  margin-right: 2px;
  margin-top: 2px;
  height: 16px;
}
[dir="rtl"] #select-all.pull-left {
  float: right !important;
  float: right;
}
.menu_icon {
  margin-right: 2px;
}
.tab-content .row {
  margin-left: 0px;
  margin-right: 0px;
}
.folder_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f114";
}
.folder_icon:before.fa-pull-left {
  margin-right: .3em;
}
.folder_icon:before.fa-pull-right {
  margin-left: .3em;
}
.folder_icon:before.pull-left {
  margin-right: .3em;
}
.folder_icon:before.pull-right {
  margin-left: .3em;
}
.notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
}
.notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.notebook_icon:before.pull-left {
  margin-right: .3em;
}
.notebook_icon:before.pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f02d";
  position: relative;
  top: -1px;
  color: #5cb85c;
}
.running_notebook_icon:before.fa-pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.fa-pull-right {
  margin-left: .3em;
}
.running_notebook_icon:before.pull-left {
  margin-right: .3em;
}
.running_notebook_icon:before.pull-right {
  margin-left: .3em;
}
.file_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f016";
  position: relative;
  top: -2px;
}
.file_icon:before.fa-pull-left {
  margin-right: .3em;
}
.file_icon:before.fa-pull-right {
  margin-left: .3em;
}
.file_icon:before.pull-left {
  margin-right: .3em;
}
.file_icon:before.pull-right {
  margin-left: .3em;
}
#notebook_toolbar .pull-right {
  padding-top: 0px;
  margin-right: -1px;
}
ul#new-menu {
  left: auto;
  right: 0;
}
#new-menu .dropdown-header {
  font-size: 10px;
  border-bottom: 1px solid #e5e5e5;
  padding: 0 0 3px;
  margin: -3px 20px 0;
}
.kernel-menu-icon {
  padding-right: 12px;
  width: 24px;
  content: "\f096";
}
.kernel-menu-icon:before {
  content: "\f096";
}
.kernel-menu-icon-current:before {
  content: "\f00c";
}
#tab_content {
  padding-top: 20px;
}
#running .panel-group .panel {
  margin-top: 3px;
  margin-bottom: 1em;
}
#running .panel-group .panel .panel-heading {
  background-color: #EEE;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 7px;
  padding-right: 7px;
  line-height: 22px;
}
#running .panel-group .panel .panel-heading a:focus,
#running .panel-group .panel .panel-heading a:hover {
  text-decoration: none;
}
#running .panel-group .panel .panel-body {
  padding: 0px;
}
#running .panel-group .panel .panel-body .list_container {
  margin-top: 0px;
  margin-bottom: 0px;
  border: 0px;
  border-radius: 0px;
}
#running .panel-group .panel .panel-body .list_container .list_item {
  border-bottom: 1px solid #ddd;
}
#running .panel-group .panel .panel-body .list_container .list_item:last-child {
  border-bottom: 0px;
}
.delete-button {
  display: none;
}
.duplicate-button {
  display: none;
}
.rename-button {
  display: none;
}
.move-button {
  display: none;
}
.download-button {
  display: none;
}
.shutdown-button {
  display: none;
}
.dynamic-instructions {
  display: inline-block;
  padding-top: 4px;
}
/*!
*
* IPython text editor webapp
*
*/
.selected-keymap i.fa {
  padding: 0px 5px;
}
.selected-keymap i.fa:before {
  content: "\f00c";
}
#mode-menu {
  overflow: auto;
  max-height: 20em;
}
.edit_app #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.edit_app #menubar .navbar {
  /* Use a negative 1 bottom margin, so the border overlaps the border of the
    header */
  margin-bottom: -1px;
}
.dirty-indicator {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator.pull-left {
  margin-right: .3em;
}
.dirty-indicator.pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-dirty.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-dirty.pull-left {
  margin-right: .3em;
}
.dirty-indicator-dirty.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  width: 20px;
}
.dirty-indicator-clean.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean.pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f00c";
}
.dirty-indicator-clean:before.fa-pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.fa-pull-right {
  margin-left: .3em;
}
.dirty-indicator-clean:before.pull-left {
  margin-right: .3em;
}
.dirty-indicator-clean:before.pull-right {
  margin-left: .3em;
}
#filename {
  font-size: 16pt;
  display: table;
  padding: 0px 5px;
}
#current-mode {
  padding-left: 5px;
  padding-right: 5px;
}
#texteditor-backdrop {
  padding-top: 20px;
  padding-bottom: 20px;
}
@media not print {
  #texteditor-backdrop {
    background-color: #EEE;
  }
}
@media print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container .CodeMirror-gutter,
  #texteditor-backdrop #texteditor-container .CodeMirror-gutters {
    background-color: #fff;
  }
}
@media not print {
  #texteditor-backdrop #texteditor-container {
    padding: 0px;
    background-color: #fff;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
.CodeMirror-dialog {
  background-color: #fff;
}
/*!
*
* IPython notebook
*
*/
/* CSS font colors for translated ANSI escape sequences */
/* The color values are a mix of
   http://www.xcolors.net/dl/baskerville-ivorylight and
   http://www.xcolors.net/dl/euphrasia */
.ansi-black-fg {
  color: #3E424D;
}
.ansi-black-bg {
  background-color: #3E424D;
}
.ansi-black-intense-fg {
  color: #282C36;
}
.ansi-black-intense-bg {
  background-color: #282C36;
}
.ansi-red-fg {
  color: #E75C58;
}
.ansi-red-bg {
  background-color: #E75C58;
}
.ansi-red-intense-fg {
  color: #B22B31;
}
.ansi-red-intense-bg {
  background-color: #B22B31;
}
.ansi-green-fg {
  color: #00A250;
}
.ansi-green-bg {
  background-color: #00A250;
}
.ansi-green-intense-fg {
  color: #007427;
}
.ansi-green-intense-bg {
  background-color: #007427;
}
.ansi-yellow-fg {
  color: #DDB62B;
}
.ansi-yellow-bg {
  background-color: #DDB62B;
}
.ansi-yellow-intense-fg {
  color: #B27D12;
}
.ansi-yellow-intense-bg {
  background-color: #B27D12;
}
.ansi-blue-fg {
  color: #208FFB;
}
.ansi-blue-bg {
  background-color: #208FFB;
}
.ansi-blue-intense-fg {
  color: #0065CA;
}
.ansi-blue-intense-bg {
  background-color: #0065CA;
}
.ansi-magenta-fg {
  color: #D160C4;
}
.ansi-magenta-bg {
  background-color: #D160C4;
}
.ansi-magenta-intense-fg {
  color: #A03196;
}
.ansi-magenta-intense-bg {
  background-color: #A03196;
}
.ansi-cyan-fg {
  color: #60C6C8;
}
.ansi-cyan-bg {
  background-color: #60C6C8;
}
.ansi-cyan-intense-fg {
  color: #258F8F;
}
.ansi-cyan-intense-bg {
  background-color: #258F8F;
}
.ansi-white-fg {
  color: #C5C1B4;
}
.ansi-white-bg {
  background-color: #C5C1B4;
}
.ansi-white-intense-fg {
  color: #A1A6B2;
}
.ansi-white-intense-bg {
  background-color: #A1A6B2;
}
.ansi-default-inverse-fg {
  color: #FFFFFF;
}
.ansi-default-inverse-bg {
  background-color: #000000;
}
.ansi-bold {
  font-weight: bold;
}
.ansi-underline {
  text-decoration: underline;
}
/* The following styles are deprecated an will be removed in a future version */
.ansibold {
  font-weight: bold;
}
.ansi-inverse {
  outline: 0.5px dotted;
}
/* use dark versions for foreground, to improve visibility */
.ansiblack {
  color: black;
}
.ansired {
  color: darkred;
}
.ansigreen {
  color: darkgreen;
}
.ansiyellow {
  color: #c4a000;
}
.ansiblue {
  color: darkblue;
}
.ansipurple {
  color: darkviolet;
}
.ansicyan {
  color: steelblue;
}
.ansigray {
  color: gray;
}
/* and light for background, for the same reason */
.ansibgblack {
  background-color: black;
}
.ansibgred {
  background-color: red;
}
.ansibggreen {
  background-color: green;
}
.ansibgyellow {
  background-color: yellow;
}
.ansibgblue {
  background-color: blue;
}
.ansibgpurple {
  background-color: magenta;
}
.ansibgcyan {
  background-color: cyan;
}
.ansibggray {
  background-color: gray;
}
div.cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  border-radius: 2px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  border-width: 1px;
  border-style: solid;
  border-color: transparent;
  width: 100%;
  padding: 5px;
  /* This acts as a spacer between cells, that is outside the border */
  margin: 0px;
  outline: none;
  position: relative;
  overflow: visible;
}
div.cell:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: transparent;
}
div.cell.jupyter-soft-selected {
  border-left-color: #E3F2FD;
  border-left-width: 1px;
  padding-left: 5px;
  border-right-color: #E3F2FD;
  border-right-width: 1px;
  background: #E3F2FD;
}
@media print {
  div.cell.jupyter-soft-selected {
    border-color: transparent;
  }
}
div.cell.selected,
div.cell.selected.jupyter-soft-selected {
  border-color: #ababab;
}
div.cell.selected:before,
div.cell.selected.jupyter-soft-selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #42A5F5;
}
@media print {
  div.cell.selected,
  div.cell.selected.jupyter-soft-selected {
    border-color: transparent;
  }
}
.edit_mode div.cell.selected {
  border-color: #66BB6A;
}
.edit_mode div.cell.selected:before {
  position: absolute;
  display: block;
  top: -1px;
  left: -1px;
  width: 5px;
  height: calc(100% +  2px);
  content: '';
  background: #66BB6A;
}
@media print {
  .edit_mode div.cell.selected {
    border-color: transparent;
  }
}
.prompt {
  /* This needs to be wide enough for 3 digit prompt numbers: In[100]: */
  min-width: 14ex;
  /* This padding is tuned to match the padding on the CodeMirror editor. */
  padding: 0.4em;
  margin: 0px;
  font-family: monospace;
  text-align: right;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
  /* Don't highlight prompt number selection */
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  /* Use default cursor */
  cursor: default;
}
@media (max-width: 540px) {
  .prompt {
    text-align: left;
  }
}
div.inner_cell {
  min-width: 0;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_area {
  border: 1px solid #cfcfcf;
  border-radius: 2px;
  background: #f7f7f7;
  line-height: 1.21429em;
}
/* This is needed so that empty prompt areas can collapse to zero height when there
   is no content in the output_subarea and the prompt. The main purpose of this is
   to make sure that empty JavaScript output_subareas have no height. */
div.prompt:empty {
  padding-top: 0;
  padding-bottom: 0;
}
div.unrecognized_cell {
  padding: 5px 5px 5px 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.unrecognized_cell .inner_cell {
  border-radius: 2px;
  padding: 5px;
  font-weight: bold;
  color: red;
  border: 1px solid #cfcfcf;
  background: #eaeaea;
}
div.unrecognized_cell .inner_cell a {
  color: inherit;
  text-decoration: none;
}
div.unrecognized_cell .inner_cell a:hover {
  color: inherit;
  text-decoration: none;
}
@media (max-width: 540px) {
  div.unrecognized_cell > div.prompt {
    display: none;
  }
}
div.code_cell {
  /* avoid page breaking on code cells when printing */
}
@media print {
  div.code_cell {
    page-break-inside: avoid;
  }
}
/* any special styling for code cells that are currently running goes here */
div.input {
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.input {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
/* input_area and input_prompt must match in top border and margin for alignment */
div.input_prompt {
  color: #303F9F;
  border-top: 1px solid transparent;
}
div.input_area > div.highlight {
  margin: 0.4em;
  border: none;
  padding: 0px;
  background-color: transparent;
}
div.input_area > div.highlight > pre {
  margin: 0px;
  border: none;
  padding: 0px;
  background-color: transparent;
}
/* The following gets added to the <head> if it is detected that the user has a
 * monospace font with inconsistent normal/bold/italic height.  See
 * notebookmain.js.  Such fonts will have keywords vertically offset with
 * respect to the rest of the text.  The user should select a better font.
 * See: https://github.com/ipython/ipython/issues/1503
 *
 * .CodeMirror span {
 *      vertical-align: bottom;
 * }
 */
.CodeMirror {
  line-height: 1.21429em;
  /* Changed from 1em to our global default */
  font-size: 14px;
  height: auto;
  /* Changed to auto to autogrow */
  background: none;
  /* Changed from white to allow our bg to show through */
}
.CodeMirror-scroll {
  /*  The CodeMirror docs are a bit fuzzy on if overflow-y should be hidden or visible.*/
  /*  We have found that if it is visible, vertical scrollbars appear with font size changes.*/
  overflow-y: hidden;
  overflow-x: auto;
}
.CodeMirror-lines {
  /* In CM2, this used to be 0.4em, but in CM3 it went to 4px. We need the em value because */
  /* we have set a different line-height and want this to scale with that. */
  /* Note that this should set vertical padding only, since CodeMirror assumes
       that horizontal padding will be set on CodeMirror pre */
  padding: 0.4em 0;
}
.CodeMirror-linenumber {
  padding: 0 8px 0 4px;
}
.CodeMirror-gutters {
  border-bottom-left-radius: 2px;
  border-top-left-radius: 2px;
}
.CodeMirror pre {
  /* In CM3 this went to 4px from 0 in CM2. This sets horizontal padding only,
    use .CodeMirror-lines for vertical */
  padding: 0 0.4em;
  border: 0;
  border-radius: 0;
}
.CodeMirror-cursor {
  border-left: 1.4px solid black;
}
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .CodeMirror-cursor {
    border-left: 2px solid black;
  }
}
@media screen and (min-width: 4320px) {
  .CodeMirror-cursor {
    border-left: 4px solid black;
  }
}
/*

Original style from softwaremaniacs.org (c) Ivan Sagalaev <Maniac@SoftwareManiacs.Org>
Adapted from GitHub theme

*/
.highlight-base {
  color: #000;
}
.highlight-variable {
  color: #000;
}
.highlight-variable-2 {
  color: #1a1a1a;
}
.highlight-variable-3 {
  color: #333333;
}
.highlight-string {
  color: #BA2121;
}
.highlight-comment {
  color: #408080;
  font-style: italic;
}
.highlight-number {
  color: #080;
}
.highlight-atom {
  color: #88F;
}
.highlight-keyword {
  color: #008000;
  font-weight: bold;
}
.highlight-builtin {
  color: #008000;
}
.highlight-error {
  color: #f00;
}
.highlight-operator {
  color: #AA22FF;
  font-weight: bold;
}
.highlight-meta {
  color: #AA22FF;
}
/* previously not defined, copying from default codemirror */
.highlight-def {
  color: #00f;
}
.highlight-string-2 {
  color: #f50;
}
.highlight-qualifier {
  color: #555;
}
.highlight-bracket {
  color: #997;
}
.highlight-tag {
  color: #170;
}
.highlight-attribute {
  color: #00c;
}
.highlight-header {
  color: blue;
}
.highlight-quote {
  color: #090;
}
.highlight-link {
  color: #00c;
}
/* apply the same style to codemirror */
.cm-s-ipython span.cm-keyword {
  color: #008000;
  font-weight: bold;
}
.cm-s-ipython span.cm-atom {
  color: #88F;
}
.cm-s-ipython span.cm-number {
  color: #080;
}
.cm-s-ipython span.cm-def {
  color: #00f;
}
.cm-s-ipython span.cm-variable {
  color: #000;
}
.cm-s-ipython span.cm-operator {
  color: #AA22FF;
  font-weight: bold;
}
.cm-s-ipython span.cm-variable-2 {
  color: #1a1a1a;
}
.cm-s-ipython span.cm-variable-3 {
  color: #333333;
}
.cm-s-ipython span.cm-comment {
  color: #408080;
  font-style: italic;
}
.cm-s-ipython span.cm-string {
  color: #BA2121;
}
.cm-s-ipython span.cm-string-2 {
  color: #f50;
}
.cm-s-ipython span.cm-meta {
  color: #AA22FF;
}
.cm-s-ipython span.cm-qualifier {
  color: #555;
}
.cm-s-ipython span.cm-builtin {
  color: #008000;
}
.cm-s-ipython span.cm-bracket {
  color: #997;
}
.cm-s-ipython span.cm-tag {
  color: #170;
}
.cm-s-ipython span.cm-attribute {
  color: #00c;
}
.cm-s-ipython span.cm-header {
  color: blue;
}
.cm-s-ipython span.cm-quote {
  color: #090;
}
.cm-s-ipython span.cm-link {
  color: #00c;
}
.cm-s-ipython span.cm-error {
  color: #f00;
}
.cm-s-ipython span.cm-tab {
  background: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAMCAYAAAAkuj5RAAAAAXNSR0IArs4c6QAAAGFJREFUSMft1LsRQFAQheHPowAKoACx3IgEKtaEHujDjORSgWTH/ZOdnZOcM/sgk/kFFWY0qV8foQwS4MKBCS3qR6ixBJvElOobYAtivseIE120FaowJPN75GMu8j/LfMwNjh4HUpwg4LUAAAAASUVORK5CYII=);
  background-position: right;
  background-repeat: no-repeat;
}
div.output_wrapper {
  /* this position must be relative to enable descendents to be absolute within it */
  position: relative;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
  z-index: 1;
}
/* class for the output area when it should be height-limited */
div.output_scroll {
  /* ideally, this would be max-height, but FF barfs all over that */
  height: 24em;
  /* FF needs this *and the wrapper* to specify full width, or it will shrinkwrap */
  width: 100%;
  overflow: auto;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
  display: block;
}
/* output div while it is collapsed */
div.output_collapsed {
  margin: 0px;
  padding: 0px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
div.out_prompt_overlay {
  height: 100%;
  padding: 0px 0.4em;
  position: absolute;
  border-radius: 2px;
}
div.out_prompt_overlay:hover {
  /* use inner shadow to get border that is computed the same on WebKit/FF */
  -webkit-box-shadow: inset 0 0 1px #000;
  box-shadow: inset 0 0 1px #000;
  background: rgba(240, 240, 240, 0.5);
}
div.output_prompt {
  color: #D84315;
}
/* This class is the outer container of all output sections. */
div.output_area {
  padding: 0px;
  page-break-inside: avoid;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
div.output_area .MathJax_Display {
  text-align: left !important;
}
div.output_area .rendered_html table {
  margin-left: 0;
  margin-right: 0;
}
div.output_area .rendered_html img {
  margin-left: 0;
  margin-right: 0;
}
div.output_area img,
div.output_area svg {
  max-width: 100%;
  height: auto;
}
div.output_area img.unconfined,
div.output_area svg.unconfined {
  max-width: none;
}
div.output_area .mglyph > img {
  max-width: none;
}
/* This is needed to protect the pre formating from global settings such
   as that of bootstrap */
.output {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: vertical;
  -moz-box-align: stretch;
  display: box;
  box-orient: vertical;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: column;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.output_area {
    /* Old browsers */
    display: -webkit-box;
    -webkit-box-orient: vertical;
    -webkit-box-align: stretch;
    display: -moz-box;
    -moz-box-orient: vertical;
    -moz-box-align: stretch;
    display: box;
    box-orient: vertical;
    box-align: stretch;
    /* Modern browsers */
    display: flex;
    flex-direction: column;
    align-items: stretch;
  }
}
div.output_area pre {
  margin: 0;
  padding: 1px 0 1px 0;
  border: 0;
  vertical-align: baseline;
  color: black;
  background-color: transparent;
  border-radius: 0;
}
/* This class is for the output subarea inside the output_area and after
   the prompt div. */
div.output_subarea {
  overflow-x: auto;
  padding: 0.4em;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
  max-width: calc(100% - 14ex);
}
div.output_scroll div.output_subarea {
  overflow-x: visible;
}
/* The rest of the output_* classes are for special styling of the different
   output types */
/* all text output has this class: */
div.output_text {
  text-align: left;
  color: #000;
  /* This has to match that of the the CodeMirror class line-height below */
  line-height: 1.21429em;
}
/* stdout/stderr are 'text' as well as 'stream', but execute_result/error are *not* streams */
div.output_stderr {
  background: #fdd;
  /* very light red background for stderr */
}
div.output_latex {
  text-align: left;
}
/* Empty output_javascript divs should have no height */
div.output_javascript:empty {
  padding: 0;
}
.js-error {
  color: darkred;
}
/* raw_input styles */
div.raw_input_container {
  line-height: 1.21429em;
  padding-top: 5px;
}
pre.raw_input_prompt {
  /* nothing needed here. */
}
input.raw_input {
  font-family: monospace;
  font-size: inherit;
  color: inherit;
  width: auto;
  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;
  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0em 0.25em;
  margin: 0em 0.25em;
}
input.raw_input:focus {
  box-shadow: none;
}
p.p-space {
  margin-bottom: 10px;
}
div.output_unrecognized {
  padding: 5px;
  font-weight: bold;
  color: red;
}
div.output_unrecognized a {
  color: inherit;
  text-decoration: none;
}
div.output_unrecognized a:hover {
  color: inherit;
  text-decoration: none;
}
.rendered_html {
  color: #000;
  /* any extras will just be numbers: */
}
.rendered_html em {
  font-style: italic;
}
.rendered_html strong {
  font-weight: bold;
}
.rendered_html u {
  text-decoration: underline;
}
.rendered_html :link {
  text-decoration: underline;
}
.rendered_html :visited {
  text-decoration: underline;
}
.rendered_html h1 {
  font-size: 185.7%;
  margin: 1.08em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h2 {
  font-size: 157.1%;
  margin: 1.27em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h3 {
  font-size: 128.6%;
  margin: 1.55em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h4 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
}
.rendered_html h5 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h6 {
  font-size: 100%;
  margin: 2em 0 0 0;
  font-weight: bold;
  line-height: 1.0;
  font-style: italic;
}
.rendered_html h1:first-child {
  margin-top: 0.538em;
}
.rendered_html h2:first-child {
  margin-top: 0.636em;
}
.rendered_html h3:first-child {
  margin-top: 0.777em;
}
.rendered_html h4:first-child {
  margin-top: 1em;
}
.rendered_html h5:first-child {
  margin-top: 1em;
}
.rendered_html h6:first-child {
  margin-top: 1em;
}
.rendered_html ul:not(.list-inline),
.rendered_html ol:not(.list-inline) {
  padding-left: 2em;
}
.rendered_html ul {
  list-style: disc;
}
.rendered_html ul ul {
  list-style: square;
  margin-top: 0;
}
.rendered_html ul ul ul {
  list-style: circle;
}
.rendered_html ol {
  list-style: decimal;
}
.rendered_html ol ol {
  list-style: upper-alpha;
  margin-top: 0;
}
.rendered_html ol ol ol {
  list-style: lower-alpha;
}
.rendered_html ol ol ol ol {
  list-style: lower-roman;
}
.rendered_html ol ol ol ol ol {
  list-style: decimal;
}
.rendered_html * + ul {
  margin-top: 1em;
}
.rendered_html * + ol {
  margin-top: 1em;
}
.rendered_html hr {
  color: black;
  background-color: black;
}
.rendered_html pre {
  margin: 1em 2em;
  padding: 0px;
  background-color: #fff;
}
.rendered_html code {
  background-color: #eff0f1;
}
.rendered_html p code {
  padding: 1px 5px;
}
.rendered_html pre code {
  background-color: #fff;
}
.rendered_html pre,
.rendered_html code {
  border: 0;
  color: #000;
  font-size: 100%;
}
.rendered_html blockquote {
  margin: 1em 2em;
}
.rendered_html table {
  margin-left: auto;
  margin-right: auto;
  border: none;
  border-collapse: collapse;
  border-spacing: 0;
  color: black;
  font-size: 12px;
  table-layout: fixed;
}
.rendered_html thead {
  border-bottom: 1px solid black;
  vertical-align: bottom;
}
.rendered_html tr,
.rendered_html th,
.rendered_html td {
  text-align: right;
  vertical-align: middle;
  padding: 0.5em 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}
.rendered_html th {
  font-weight: bold;
}
.rendered_html tbody tr:nth-child(odd) {
  background: #f5f5f5;
}
.rendered_html tbody tr:hover {
  background: rgba(66, 165, 245, 0.2);
}
.rendered_html * + table {
  margin-top: 1em;
}
.rendered_html p {
  text-align: left;
}
.rendered_html * + p {
  margin-top: 1em;
}
.rendered_html img {
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.rendered_html * + img {
  margin-top: 1em;
}
.rendered_html img,
.rendered_html svg {
  max-width: 100%;
  height: auto;
}
.rendered_html img.unconfined,
.rendered_html svg.unconfined {
  max-width: none;
}
.rendered_html .alert {
  margin-bottom: initial;
}
.rendered_html * + .alert {
  margin-top: 1em;
}
[dir="rtl"] .rendered_html p {
  text-align: right;
}
div.text_cell {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
}
@media (max-width: 540px) {
  div.text_cell > div.prompt {
    display: none;
  }
}
div.text_cell_render {
  /*font-family: "Helvetica Neue", Arial, Helvetica, Geneva, sans-serif;*/
  outline: none;
  resize: none;
  width: inherit;
  border-style: none;
  padding: 0.5em 0.5em 0.5em 0.4em;
  color: #000;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
a.anchor-link:link {
  text-decoration: none;
  padding: 0px 20px;
  visibility: hidden;
}
h1:hover .anchor-link,
h2:hover .anchor-link,
h3:hover .anchor-link,
h4:hover .anchor-link,
h5:hover .anchor-link,
h6:hover .anchor-link {
  visibility: visible;
}
.text_cell.rendered .input_area {
  display: none;
}
.text_cell.rendered .rendered_html {
  overflow-x: auto;
  overflow-y: hidden;
}
.text_cell.rendered .rendered_html tr,
.text_cell.rendered .rendered_html th,
.text_cell.rendered .rendered_html td {
  max-width: none;
}
.text_cell.unrendered .text_cell_render {
  display: none;
}
.text_cell .dropzone .input_area {
  border: 2px dashed #bababa;
  margin: -1px;
}
.cm-header-1,
.cm-header-2,
.cm-header-3,
.cm-header-4,
.cm-header-5,
.cm-header-6 {
  font-weight: bold;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.cm-header-1 {
  font-size: 185.7%;
}
.cm-header-2 {
  font-size: 157.1%;
}
.cm-header-3 {
  font-size: 128.6%;
}
.cm-header-4 {
  font-size: 110%;
}
.cm-header-5 {
  font-size: 100%;
  font-style: italic;
}
.cm-header-6 {
  font-size: 100%;
  font-style: italic;
}
/*!
*
* IPython notebook webapp
*
*/
@media (max-width: 767px) {
  .notebook_app {
    padding-left: 0px;
    padding-right: 0px;
  }
}
#ipython-main-app {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook_panel {
  margin: 0px;
  padding: 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  height: 100%;
}
div#notebook {
  font-size: 14px;
  line-height: 20px;
  overflow-y: hidden;
  overflow-x: auto;
  width: 100%;
  /* This spaces the page away from the edge of the notebook area */
  padding-top: 20px;
  margin: 0px;
  outline: none;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  min-height: 100%;
}
@media not print {
  #notebook-container {
    padding: 15px;
    background-color: #fff;
    min-height: 0;
    -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
    box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  }
}
@media print {
  #notebook-container {
    width: 100%;
  }
}
div.ui-widget-content {
  border: 1px solid #ababab;
  outline: none;
}
pre.dialog {
  background-color: #f7f7f7;
  border: 1px solid #ddd;
  border-radius: 2px;
  padding: 0.4em;
  padding-left: 2em;
}
p.dialog {
  padding: 0.2em;
}
/* Word-wrap output correctly.  This is the CSS3 spelling, though Firefox seems
   to not honor it correctly.  Webkit browsers (Chrome, rekonq, Safari) do.
 */
pre,
code,
kbd,
samp {
  white-space: pre-wrap;
}
#fonttest {
  font-family: monospace;
}
p {
  margin-bottom: 0;
}
.end_space {
  min-height: 100px;
  transition: height .2s ease;
}
.notebook_app > #header {
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
@media not print {
  .notebook_app {
    background-color: #EEE;
  }
}
kbd {
  border-style: solid;
  border-width: 1px;
  box-shadow: none;
  margin: 2px;
  padding-left: 2px;
  padding-right: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
.jupyter-keybindings {
  padding: 1px;
  line-height: 24px;
  border-bottom: 1px solid gray;
}
.jupyter-keybindings input {
  margin: 0;
  padding: 0;
  border: none;
}
.jupyter-keybindings i {
  padding: 6px;
}
.well code {
  background-color: #ffffff;
  border-color: #ababab;
  border-width: 1px;
  border-style: solid;
  padding: 2px;
  padding-top: 1px;
  padding-bottom: 1px;
}
/* CSS for the cell toolbar */
.celltoolbar {
  border: thin solid #CFCFCF;
  border-bottom: none;
  background: #EEE;
  border-radius: 2px 2px 0px 0px;
  width: 100%;
  height: 29px;
  padding-right: 4px;
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  /* Old browsers */
  -webkit-box-pack: end;
  -moz-box-pack: end;
  box-pack: end;
  /* Modern browsers */
  justify-content: flex-end;
  display: -webkit-flex;
}
@media print {
  .celltoolbar {
    display: none;
  }
}
.ctb_hideshow {
  display: none;
  vertical-align: bottom;
}
/* ctb_show is added to the ctb_hideshow div to show the cell toolbar.
   Cell toolbars are only shown when the ctb_global_show class is also set.
*/
.ctb_global_show .ctb_show.ctb_hideshow {
  display: block;
}
.ctb_global_show .ctb_show + .input_area,
.ctb_global_show .ctb_show + div.text_cell_input,
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border-top-right-radius: 0px;
  border-top-left-radius: 0px;
}
.ctb_global_show .ctb_show ~ div.text_cell_render {
  border: 1px solid #cfcfcf;
}
.celltoolbar {
  font-size: 87%;
  padding-top: 3px;
}
.celltoolbar select {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  width: inherit;
  font-size: inherit;
  height: 22px;
  padding: 0px;
  display: inline-block;
}
.celltoolbar select:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.celltoolbar select::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.celltoolbar select:-ms-input-placeholder {
  color: #999;
}
.celltoolbar select::-webkit-input-placeholder {
  color: #999;
}
.celltoolbar select::-ms-expand {
  border: 0;
  background-color: transparent;
}
.celltoolbar select[disabled],
.celltoolbar select[readonly],
fieldset[disabled] .celltoolbar select {
  background-color: #eeeeee;
  opacity: 1;
}
.celltoolbar select[disabled],
fieldset[disabled] .celltoolbar select {
  cursor: not-allowed;
}
textarea.celltoolbar select {
  height: auto;
}
select.celltoolbar select {
  height: 30px;
  line-height: 30px;
}
textarea.celltoolbar select,
select[multiple].celltoolbar select {
  height: auto;
}
.celltoolbar label {
  margin-left: 5px;
  margin-right: 5px;
}
.tags_button_container {
  width: 100%;
  display: flex;
}
.tag-container {
  display: flex;
  flex-direction: row;
  flex-grow: 1;
  overflow: hidden;
  position: relative;
}
.tag-container > * {
  margin: 0 4px;
}
.remove-tag-btn {
  margin-left: 4px;
}
.tags-input {
  display: flex;
}
.cell-tag:last-child:after {
  content: "";
  position: absolute;
  right: 0;
  width: 40px;
  height: 100%;
  /* Fade to background color of cell toolbar */
  background: linear-gradient(to right, rgba(0, 0, 0, 0), #EEE);
}
.tags-input > * {
  margin-left: 4px;
}
.cell-tag,
.tags-input input,
.tags-input button {
  display: block;
  width: 100%;
  height: 32px;
  padding: 6px 12px;
  font-size: 13px;
  line-height: 1.42857143;
  color: #555555;
  background-color: #fff;
  background-image: none;
  border: 1px solid #ccc;
  border-radius: 2px;
  -webkit-box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  box-shadow: inset 0 1px 1px rgba(0, 0, 0, 0.075);
  -webkit-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  -o-transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  transition: border-color ease-in-out .15s, box-shadow ease-in-out .15s;
  height: 30px;
  padding: 5px 10px;
  font-size: 12px;
  line-height: 1.5;
  border-radius: 1px;
  box-shadow: none;
  width: inherit;
  font-size: inherit;
  height: 22px;
  line-height: 22px;
  padding: 0px 4px;
  display: inline-block;
}
.cell-tag:focus,
.tags-input input:focus,
.tags-input button:focus {
  border-color: #66afe9;
  outline: 0;
  -webkit-box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
  box-shadow: inset 0 1px 1px rgba(0,0,0,.075), 0 0 8px rgba(102, 175, 233, 0.6);
}
.cell-tag::-moz-placeholder,
.tags-input input::-moz-placeholder,
.tags-input button::-moz-placeholder {
  color: #999;
  opacity: 1;
}
.cell-tag:-ms-input-placeholder,
.tags-input input:-ms-input-placeholder,
.tags-input button:-ms-input-placeholder {
  color: #999;
}
.cell-tag::-webkit-input-placeholder,
.tags-input input::-webkit-input-placeholder,
.tags-input button::-webkit-input-placeholder {
  color: #999;
}
.cell-tag::-ms-expand,
.tags-input input::-ms-expand,
.tags-input button::-ms-expand {
  border: 0;
  background-color: transparent;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
.cell-tag[readonly],
.tags-input input[readonly],
.tags-input button[readonly],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  background-color: #eeeeee;
  opacity: 1;
}
.cell-tag[disabled],
.tags-input input[disabled],
.tags-input button[disabled],
fieldset[disabled] .cell-tag,
fieldset[disabled] .tags-input input,
fieldset[disabled] .tags-input button {
  cursor: not-allowed;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button {
  height: auto;
}
select.cell-tag,
select.tags-input input,
select.tags-input button {
  height: 30px;
  line-height: 30px;
}
textarea.cell-tag,
textarea.tags-input input,
textarea.tags-input button,
select[multiple].cell-tag,
select[multiple].tags-input input,
select[multiple].tags-input button {
  height: auto;
}
.cell-tag,
.tags-input button {
  padding: 0px 4px;
}
.cell-tag {
  background-color: #fff;
  white-space: nowrap;
}
.tags-input input[type=text]:focus {
  outline: none;
  box-shadow: none;
  border-color: #ccc;
}
.completions {
  position: absolute;
  z-index: 110;
  overflow: hidden;
  border: 1px solid #ababab;
  border-radius: 2px;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  line-height: 1;
}
.completions select {
  background: white;
  outline: none;
  border: none;
  padding: 0px;
  margin: 0px;
  overflow: auto;
  font-family: monospace;
  font-size: 110%;
  color: #000;
  width: auto;
}
.completions select option.context {
  color: #286090;
}
#kernel_logo_widget .current_kernel_logo {
  display: none;
  margin-top: -1px;
  margin-bottom: -1px;
  width: 32px;
  height: 32px;
}
[dir="rtl"] #kernel_logo_widget {
  float: left !important;
  float: left;
}
.modal .modal-body .move-path {
  display: flex;
  flex-direction: row;
  justify-content: space;
  align-items: center;
}
.modal .modal-body .move-path .server-root {
  padding-right: 20px;
}
.modal .modal-body .move-path .path-input {
  flex: 1;
}
#menubar {
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
  margin-top: 1px;
}
#menubar .navbar {
  border-top: 1px;
  border-radius: 0px 0px 2px 2px;
  margin-bottom: 0px;
}
#menubar .navbar-toggle {
  float: left;
  padding-top: 7px;
  padding-bottom: 7px;
  border: none;
}
#menubar .navbar-collapse {
  clear: left;
}
[dir="rtl"] #menubar .navbar-toggle {
  float: right;
}
[dir="rtl"] #menubar .navbar-collapse {
  clear: right;
}
[dir="rtl"] #menubar .navbar-nav {
  float: right;
}
[dir="rtl"] #menubar .nav {
  padding-right: 0px;
}
[dir="rtl"] #menubar .navbar-nav > li {
  float: right;
}
[dir="rtl"] #menubar .navbar-right {
  float: left !important;
}
[dir="rtl"] ul.dropdown-menu {
  text-align: right;
  left: auto;
}
[dir="rtl"] ul#new-menu.dropdown-menu {
  right: auto;
  left: 0;
}
.nav-wrapper {
  border-bottom: 1px solid #e7e7e7;
}
i.menu-icon {
  padding-top: 4px;
}
[dir="rtl"] i.menu-icon.pull-right {
  float: left !important;
  float: left;
}
ul#help_menu li a {
  overflow: hidden;
  padding-right: 2.2em;
}
ul#help_menu li a i {
  margin-right: -1.2em;
}
[dir="rtl"] ul#help_menu li a {
  padding-left: 2.2em;
}
[dir="rtl"] ul#help_menu li a i {
  margin-right: 0;
  margin-left: -1.2em;
}
[dir="rtl"] ul#help_menu li a i.pull-right {
  float: left !important;
  float: left;
}
.dropdown-submenu {
  position: relative;
}
.dropdown-submenu > .dropdown-menu {
  top: 0;
  left: 100%;
  margin-top: -6px;
  margin-left: -1px;
}
[dir="rtl"] .dropdown-submenu > .dropdown-menu {
  right: 100%;
  margin-right: -1px;
}
.dropdown-submenu:hover > .dropdown-menu {
  display: block;
}
.dropdown-submenu > a:after {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  display: block;
  content: "\f0da";
  float: right;
  color: #333333;
  margin-top: 2px;
  margin-right: -10px;
}
.dropdown-submenu > a:after.fa-pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.fa-pull-right {
  margin-left: .3em;
}
.dropdown-submenu > a:after.pull-left {
  margin-right: .3em;
}
.dropdown-submenu > a:after.pull-right {
  margin-left: .3em;
}
[dir="rtl"] .dropdown-submenu > a:after {
  float: left;
  content: "\f0d9";
  margin-right: 0;
  margin-left: -10px;
}
.dropdown-submenu:hover > a:after {
  color: #262626;
}
.dropdown-submenu.pull-left {
  float: none;
}
.dropdown-submenu.pull-left > .dropdown-menu {
  left: -100%;
  margin-left: 10px;
}
#notification_area {
  float: right !important;
  float: right;
  z-index: 10;
}
[dir="rtl"] #notification_area {
  float: left !important;
  float: left;
}
.indicator_area {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] .indicator_area {
  float: left !important;
  float: left;
}
#kernel_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  border-left: 1px solid;
}
#kernel_indicator .kernel_indicator_name {
  padding-left: 5px;
  padding-right: 5px;
}
[dir="rtl"] #kernel_indicator {
  float: left !important;
  float: left;
  border-left: 0;
  border-right: 1px solid;
}
#modal_indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
}
[dir="rtl"] #modal_indicator {
  float: left !important;
  float: left;
}
#readonly-indicator {
  float: right !important;
  float: right;
  color: #777;
  margin-left: 5px;
  margin-right: 5px;
  width: 11px;
  z-index: 10;
  text-align: center;
  width: auto;
  margin-top: 2px;
  margin-bottom: 0px;
  margin-left: 0px;
  margin-right: 0px;
  display: none;
}
.modal_indicator:before {
  width: 1.28571429em;
  text-align: center;
}
.edit_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f040";
}
.edit_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.edit_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.edit_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: ' ';
}
.command_mode .modal_indicator:before.fa-pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.fa-pull-right {
  margin-left: .3em;
}
.command_mode .modal_indicator:before.pull-left {
  margin-right: .3em;
}
.command_mode .modal_indicator:before.pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f10c";
}
.kernel_idle_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_idle_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_idle_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f111";
}
.kernel_busy_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_busy_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_busy_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f1e2";
}
.kernel_dead_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_dead_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_dead_icon:before.pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before {
  display: inline-block;
  font: normal normal normal 14px/1 FontAwesome;
  font-size: inherit;
  text-rendering: auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  content: "\f127";
}
.kernel_disconnected_icon:before.fa-pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.fa-pull-right {
  margin-left: .3em;
}
.kernel_disconnected_icon:before.pull-left {
  margin-right: .3em;
}
.kernel_disconnected_icon:before.pull-right {
  margin-left: .3em;
}
.notification_widget {
  color: #777;
  z-index: 10;
  background: rgba(240, 240, 240, 0.5);
  margin-right: 4px;
  color: #333;
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget:focus,
.notification_widget.focus {
  color: #333;
  background-color: #e6e6e6;
  border-color: #8c8c8c;
}
.notification_widget:hover {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  color: #333;
  background-color: #e6e6e6;
  border-color: #adadad;
}
.notification_widget:active:hover,
.notification_widget.active:hover,
.open > .dropdown-toggle.notification_widget:hover,
.notification_widget:active:focus,
.notification_widget.active:focus,
.open > .dropdown-toggle.notification_widget:focus,
.notification_widget:active.focus,
.notification_widget.active.focus,
.open > .dropdown-toggle.notification_widget.focus {
  color: #333;
  background-color: #d4d4d4;
  border-color: #8c8c8c;
}
.notification_widget:active,
.notification_widget.active,
.open > .dropdown-toggle.notification_widget {
  background-image: none;
}
.notification_widget.disabled:hover,
.notification_widget[disabled]:hover,
fieldset[disabled] .notification_widget:hover,
.notification_widget.disabled:focus,
.notification_widget[disabled]:focus,
fieldset[disabled] .notification_widget:focus,
.notification_widget.disabled.focus,
.notification_widget[disabled].focus,
fieldset[disabled] .notification_widget.focus {
  background-color: #fff;
  border-color: #ccc;
}
.notification_widget .badge {
  color: #fff;
  background-color: #333;
}
.notification_widget.warning {
  color: #fff;
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning:focus,
.notification_widget.warning.focus {
  color: #fff;
  background-color: #ec971f;
  border-color: #985f0d;
}
.notification_widget.warning:hover {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  color: #fff;
  background-color: #ec971f;
  border-color: #d58512;
}
.notification_widget.warning:active:hover,
.notification_widget.warning.active:hover,
.open > .dropdown-toggle.notification_widget.warning:hover,
.notification_widget.warning:active:focus,
.notification_widget.warning.active:focus,
.open > .dropdown-toggle.notification_widget.warning:focus,
.notification_widget.warning:active.focus,
.notification_widget.warning.active.focus,
.open > .dropdown-toggle.notification_widget.warning.focus {
  color: #fff;
  background-color: #d58512;
  border-color: #985f0d;
}
.notification_widget.warning:active,
.notification_widget.warning.active,
.open > .dropdown-toggle.notification_widget.warning {
  background-image: none;
}
.notification_widget.warning.disabled:hover,
.notification_widget.warning[disabled]:hover,
fieldset[disabled] .notification_widget.warning:hover,
.notification_widget.warning.disabled:focus,
.notification_widget.warning[disabled]:focus,
fieldset[disabled] .notification_widget.warning:focus,
.notification_widget.warning.disabled.focus,
.notification_widget.warning[disabled].focus,
fieldset[disabled] .notification_widget.warning.focus {
  background-color: #f0ad4e;
  border-color: #eea236;
}
.notification_widget.warning .badge {
  color: #f0ad4e;
  background-color: #fff;
}
.notification_widget.success {
  color: #fff;
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success:focus,
.notification_widget.success.focus {
  color: #fff;
  background-color: #449d44;
  border-color: #255625;
}
.notification_widget.success:hover {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  color: #fff;
  background-color: #449d44;
  border-color: #398439;
}
.notification_widget.success:active:hover,
.notification_widget.success.active:hover,
.open > .dropdown-toggle.notification_widget.success:hover,
.notification_widget.success:active:focus,
.notification_widget.success.active:focus,
.open > .dropdown-toggle.notification_widget.success:focus,
.notification_widget.success:active.focus,
.notification_widget.success.active.focus,
.open > .dropdown-toggle.notification_widget.success.focus {
  color: #fff;
  background-color: #398439;
  border-color: #255625;
}
.notification_widget.success:active,
.notification_widget.success.active,
.open > .dropdown-toggle.notification_widget.success {
  background-image: none;
}
.notification_widget.success.disabled:hover,
.notification_widget.success[disabled]:hover,
fieldset[disabled] .notification_widget.success:hover,
.notification_widget.success.disabled:focus,
.notification_widget.success[disabled]:focus,
fieldset[disabled] .notification_widget.success:focus,
.notification_widget.success.disabled.focus,
.notification_widget.success[disabled].focus,
fieldset[disabled] .notification_widget.success.focus {
  background-color: #5cb85c;
  border-color: #4cae4c;
}
.notification_widget.success .badge {
  color: #5cb85c;
  background-color: #fff;
}
.notification_widget.info {
  color: #fff;
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info:focus,
.notification_widget.info.focus {
  color: #fff;
  background-color: #31b0d5;
  border-color: #1b6d85;
}
.notification_widget.info:hover {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  color: #fff;
  background-color: #31b0d5;
  border-color: #269abc;
}
.notification_widget.info:active:hover,
.notification_widget.info.active:hover,
.open > .dropdown-toggle.notification_widget.info:hover,
.notification_widget.info:active:focus,
.notification_widget.info.active:focus,
.open > .dropdown-toggle.notification_widget.info:focus,
.notification_widget.info:active.focus,
.notification_widget.info.active.focus,
.open > .dropdown-toggle.notification_widget.info.focus {
  color: #fff;
  background-color: #269abc;
  border-color: #1b6d85;
}
.notification_widget.info:active,
.notification_widget.info.active,
.open > .dropdown-toggle.notification_widget.info {
  background-image: none;
}
.notification_widget.info.disabled:hover,
.notification_widget.info[disabled]:hover,
fieldset[disabled] .notification_widget.info:hover,
.notification_widget.info.disabled:focus,
.notification_widget.info[disabled]:focus,
fieldset[disabled] .notification_widget.info:focus,
.notification_widget.info.disabled.focus,
.notification_widget.info[disabled].focus,
fieldset[disabled] .notification_widget.info.focus {
  background-color: #5bc0de;
  border-color: #46b8da;
}
.notification_widget.info .badge {
  color: #5bc0de;
  background-color: #fff;
}
.notification_widget.danger {
  color: #fff;
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger:focus,
.notification_widget.danger.focus {
  color: #fff;
  background-color: #c9302c;
  border-color: #761c19;
}
.notification_widget.danger:hover {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  color: #fff;
  background-color: #c9302c;
  border-color: #ac2925;
}
.notification_widget.danger:active:hover,
.notification_widget.danger.active:hover,
.open > .dropdown-toggle.notification_widget.danger:hover,
.notification_widget.danger:active:focus,
.notification_widget.danger.active:focus,
.open > .dropdown-toggle.notification_widget.danger:focus,
.notification_widget.danger:active.focus,
.notification_widget.danger.active.focus,
.open > .dropdown-toggle.notification_widget.danger.focus {
  color: #fff;
  background-color: #ac2925;
  border-color: #761c19;
}
.notification_widget.danger:active,
.notification_widget.danger.active,
.open > .dropdown-toggle.notification_widget.danger {
  background-image: none;
}
.notification_widget.danger.disabled:hover,
.notification_widget.danger[disabled]:hover,
fieldset[disabled] .notification_widget.danger:hover,
.notification_widget.danger.disabled:focus,
.notification_widget.danger[disabled]:focus,
fieldset[disabled] .notification_widget.danger:focus,
.notification_widget.danger.disabled.focus,
.notification_widget.danger[disabled].focus,
fieldset[disabled] .notification_widget.danger.focus {
  background-color: #d9534f;
  border-color: #d43f3a;
}
.notification_widget.danger .badge {
  color: #d9534f;
  background-color: #fff;
}
div#pager {
  background-color: #fff;
  font-size: 14px;
  line-height: 20px;
  overflow: hidden;
  display: none;
  position: fixed;
  bottom: 0px;
  width: 100%;
  max-height: 50%;
  padding-top: 8px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  /* Display over codemirror */
  z-index: 100;
  /* Hack which prevents jquery ui resizable from changing top. */
  top: auto !important;
}
div#pager pre {
  line-height: 1.21429em;
  color: #000;
  background-color: #f7f7f7;
  padding: 0.4em;
}
div#pager #pager-button-area {
  position: absolute;
  top: 8px;
  right: 20px;
}
div#pager #pager-contents {
  position: relative;
  overflow: auto;
  width: 100%;
  height: 100%;
}
div#pager #pager-contents #pager-container {
  position: relative;
  padding: 15px 0px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
div#pager .ui-resizable-handle {
  top: 0px;
  height: 8px;
  background: #f7f7f7;
  border-top: 1px solid #cfcfcf;
  border-bottom: 1px solid #cfcfcf;
  /* This injects handle bars (a short, wide = symbol) for 
        the resize handle. */
}
div#pager .ui-resizable-handle::after {
  content: '';
  top: 2px;
  left: 50%;
  height: 3px;
  width: 30px;
  margin-left: -15px;
  position: absolute;
  border-top: 1px solid #cfcfcf;
}
.quickhelp {
  /* Old browsers */
  display: -webkit-box;
  -webkit-box-orient: horizontal;
  -webkit-box-align: stretch;
  display: -moz-box;
  -moz-box-orient: horizontal;
  -moz-box-align: stretch;
  display: box;
  box-orient: horizontal;
  box-align: stretch;
  /* Modern browsers */
  display: flex;
  flex-direction: row;
  align-items: stretch;
  line-height: 1.8em;
}
.shortcut_key {
  display: inline-block;
  width: 21ex;
  text-align: right;
  font-family: monospace;
}
.shortcut_descr {
  display: inline-block;
  /* Old browsers */
  -webkit-box-flex: 1;
  -moz-box-flex: 1;
  box-flex: 1;
  /* Modern browsers */
  flex: 1;
}
span.save_widget {
  height: 30px;
  margin-top: 4px;
  display: flex;
  justify-content: flex-start;
  align-items: baseline;
  width: 50%;
  flex: 1;
}
span.save_widget span.filename {
  height: 100%;
  line-height: 1em;
  margin-left: 16px;
  border: none;
  font-size: 146.5%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
  border-radius: 2px;
}
span.save_widget span.filename:hover {
  background-color: #e6e6e6;
}
[dir="rtl"] span.save_widget.pull-left {
  float: right !important;
  float: right;
}
[dir="rtl"] span.save_widget span.filename {
  margin-left: 0;
  margin-right: 16px;
}
span.checkpoint_status,
span.autosave_status {
  font-size: small;
  white-space: nowrap;
  padding: 0 5px;
}
@media (max-width: 767px) {
  span.save_widget {
    font-size: small;
    padding: 0 0 0 5px;
  }
  span.checkpoint_status,
  span.autosave_status {
    display: none;
  }
}
@media (min-width: 768px) and (max-width: 991px) {
  span.checkpoint_status {
    display: none;
  }
  span.autosave_status {
    font-size: x-small;
  }
}
.toolbar {
  padding: 0px;
  margin-left: -5px;
  margin-top: 2px;
  margin-bottom: 5px;
  box-sizing: border-box;
  -moz-box-sizing: border-box;
  -webkit-box-sizing: border-box;
}
.toolbar select,
.toolbar label {
  width: auto;
  vertical-align: middle;
  margin-right: 2px;
  margin-bottom: 0px;
  display: inline;
  font-size: 92%;
  margin-left: 0.3em;
  margin-right: 0.3em;
  padding: 0px;
  padding-top: 3px;
}
.toolbar .btn {
  padding: 2px 8px;
}
.toolbar .btn-group {
  margin-top: 0px;
  margin-left: 5px;
}
.toolbar-btn-label {
  margin-left: 6px;
}
#maintoolbar {
  margin-bottom: -3px;
  margin-top: -8px;
  border: 0px;
  min-height: 27px;
  margin-left: 0px;
  padding-top: 11px;
  padding-bottom: 3px;
}
#maintoolbar .navbar-text {
  float: none;
  vertical-align: middle;
  text-align: right;
  margin-left: 5px;
  margin-right: 0px;
  margin-top: 0px;
}
.select-xs {
  height: 24px;
}
[dir="rtl"] .btn-group > .btn,
.btn-group-vertical > .btn {
  float: right;
}
.pulse,
.dropdown-menu > li > a.pulse,
li.pulse > a.dropdown-toggle,
li.pulse.open > a.dropdown-toggle {
  background-color: #F37626;
  color: white;
}
/**
 * Primary styles
 *
 * Author: Jupyter Development Team
 */
/** WARNING IF YOU ARE EDITTING THIS FILE, if this is a .css file, It has a lot
 * of chance of beeing generated from the ../less/[samename].less file, you can
 * try to get back the less file by reverting somme commit in history
 **/
/*
 * We'll try to get something pretty, so we
 * have some strange css to have the scroll bar on
 * the left with fix button on the top right of the tooltip
 */
@-moz-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-webkit-keyframes fadeOut {
  from {
    opacity: 1;
  }
  to {
    opacity: 0;
  }
}
@-moz-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}
/*properties of tooltip after "expand"*/
.bigtooltip {
  overflow: auto;
  height: 200px;
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
}
/*properties of tooltip before "expand"*/
.smalltooltip {
  -webkit-transition-property: height;
  -webkit-transition-duration: 500ms;
  -moz-transition-property: height;
  -moz-transition-duration: 500ms;
  transition-property: height;
  transition-duration: 500ms;
  text-overflow: ellipsis;
  overflow: hidden;
  height: 80px;
}
.tooltipbuttons {
  position: absolute;
  padding-right: 15px;
  top: 0px;
  right: 0px;
}
.tooltiptext {
  /*avoid the button to overlap on some docstring*/
  padding-right: 30px;
}
.ipython_tooltip {
  max-width: 700px;
  /*fade-in animation when inserted*/
  -webkit-animation: fadeOut 400ms;
  -moz-animation: fadeOut 400ms;
  animation: fadeOut 400ms;
  -webkit-animation: fadeIn 400ms;
  -moz-animation: fadeIn 400ms;
  animation: fadeIn 400ms;
  vertical-align: middle;
  background-color: #f7f7f7;
  overflow: visible;
  border: #ababab 1px solid;
  outline: none;
  padding: 3px;
  margin: 0px;
  padding-left: 7px;
  font-family: monospace;
  min-height: 50px;
  -moz-box-shadow: 0px 6px 10px -1px #adadad;
  -webkit-box-shadow: 0px 6px 10px -1px #adadad;
  box-shadow: 0px 6px 10px -1px #adadad;
  border-radius: 2px;
  position: absolute;
  z-index: 1000;
}
.ipython_tooltip a {
  float: right;
}
.ipython_tooltip .tooltiptext pre {
  border: 0;
  border-radius: 0;
  font-size: 100%;
  background-color: #f7f7f7;
}
.pretooltiparrow {
  left: 0px;
  margin: 0px;
  top: -16px;
  width: 40px;
  height: 16px;
  overflow: hidden;
  position: absolute;
}
.pretooltiparrow:before {
  background-color: #f7f7f7;
  border: 1px #ababab solid;
  z-index: 11;
  content: "";
  position: absolute;
  left: 15px;
  top: 10px;
  width: 25px;
  height: 25px;
  -webkit-transform: rotate(45deg);
  -moz-transform: rotate(45deg);
  -ms-transform: rotate(45deg);
  -o-transform: rotate(45deg);
}
ul.typeahead-list i {
  margin-left: -10px;
  width: 18px;
}
[dir="rtl"] ul.typeahead-list i {
  margin-left: 0;
  margin-right: -10px;
}
ul.typeahead-list {
  max-height: 80vh;
  overflow: auto;
}
ul.typeahead-list > li > a {
  /** Firefox bug **/
  /* see https://github.com/jupyter/notebook/issues/559 */
  white-space: normal;
}
ul.typeahead-list  > li > a.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .typeahead-list {
  text-align: right;
}
.cmd-palette .modal-body {
  padding: 7px;
}
.cmd-palette form {
  background: white;
}
.cmd-palette input {
  outline: none;
}
.no-shortcut {
  min-width: 20px;
  color: transparent;
}
[dir="rtl"] .no-shortcut.pull-right {
  float: left !important;
  float: left;
}
[dir="rtl"] .command-shortcut.pull-right {
  float: left !important;
  float: left;
}
.command-shortcut:before {
  content: "(command mode)";
  padding-right: 3px;
  color: #777777;
}
.edit-shortcut:before {
  content: "(edit)";
  padding-right: 3px;
  color: #777777;
}
[dir="rtl"] .edit-shortcut.pull-right {
  float: left !important;
  float: left;
}
#find-and-replace #replace-preview .match,
#find-and-replace #replace-preview .insert {
  background-color: #BBDEFB;
  border-color: #90CAF9;
  border-style: solid;
  border-width: 1px;
  border-radius: 0px;
}
[dir="ltr"] #find-and-replace .input-group-btn + .form-control {
  border-left: none;
}
[dir="rtl"] #find-and-replace .input-group-btn + .form-control {
  border-right: none;
}
#find-and-replace #replace-preview .replace .match {
  background-color: #FFCDD2;
  border-color: #EF9A9A;
  border-radius: 0px;
}
#find-and-replace #replace-preview .replace .insert {
  background-color: #C8E6C9;
  border-color: #A5D6A7;
  border-radius: 0px;
}
#find-and-replace #replace-preview {
  max-height: 60vh;
  overflow: auto;
}
#find-and-replace #replace-preview pre {
  padding: 5px 10px;
}
.terminal-app {
  background: #EEE;
}
.terminal-app #header {
  background: #fff;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.2);
}
.terminal-app .terminal {
  width: 100%;
  float: left;
  font-family: monospace;
  color: white;
  background: black;
  padding: 0.4em;
  border-radius: 2px;
  -webkit-box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
  box-shadow: 0px 0px 12px 1px rgba(87, 87, 87, 0.4);
}
.terminal-app .terminal,
.terminal-app .terminal dummy-screen {
  line-height: 1em;
  font-size: 14px;
}
.terminal-app .terminal .xterm-rows {
  padding: 10px;
}
.terminal-app .terminal-cursor {
  color: black;
  background: white;
}
.terminal-app #terminado-container {
  margin-top: 20px;
}
/*# sourceMappingURL=style.min.css.map */
    </style>
<style type="text/css">
    .highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */
    </style>


<style type="text/css">
/* Overrides of notebook CSS for static HTML export */
body {
  overflow: visible;
  padding: 8px;
}

div#notebook {
  overflow: visible;
  border-top: none;
}@media print {
  div.cell {
    display: block;
    page-break-inside: avoid;
  } 
  div.output_wrapper { 
    display: block;
    page-break-inside: avoid; 
  }
  div.output { 
    display: block;
    page-break-inside: avoid; 
  }
}
</style>

<!-- Custom stylesheet, it must be in the same directory as the html file -->
<link rel="stylesheet" href="custom.css">

<!-- Loading mathjax macro -->
<!-- Load mathjax -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-AMS_HTML"></script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [ ['$','$'], ["\\(","\\)"] ],
            displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
            processEscapes: true,
            processEnvironments: true
        },
        // Center justify equations in code and markdown cells. Elsewhere
        // we use CSS to left justify single line equations in code cells.
        displayAlign: 'center',
        "HTML-CSS": {
            styles: {'.MathJax_Display': {"margin": 0}},
            linebreaks: { automatic: true }
        }
    });
    </script>
    <!-- End of mathjax configuration --></head>
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h1 id="Titanic-ML-Competition">Titanic ML Competition<a class="anchor-link" href="#Titanic-ML-Competition">&#182;</a></h1>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Introduction">Introduction<a class="anchor-link" href="#Introduction">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In this notebook we will be exploring the data for the Titanic machine learning competition from <a href="https://www.kaggle.com/c/titanic/overview">Kaggle</a>. The goals of the notebook are to:</p>
<ul>
<li>Better understand the data.</li>
<li>See data relationships.</li>
<li>Determine if there are patterns within the data.</li>
<li>See what sorts of people were more likely to survive the disaster.</li>
</ul>
<p>After this, we will create ML models to make predictions.</p>
<p><em>Introduction from Kaggle</em></p>

<pre><code>The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).</code></pre>
<center>
    <img src='https://miro.medium.com/max/2000/1*fBkTkunRJ88FdEXEcGU_fg.jpeg' heigh=400 width=400>
</center>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Imports">Imports<a class="anchor-link" href="#Imports">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[1]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>
<span class="kn">import</span> <span class="nn">squarify</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[2]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">math</span> <span class="kn">import</span> <span class="n">pi</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[3]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">preprocessing</span> <span class="kn">import</span> <span class="n">encodeDataset</span>
<span class="kn">from</span> <span class="nn">preprocessing</span> <span class="kn">import</span> <span class="n">encodeAndNormalizeData</span>
<span class="kn">from</span> <span class="nn">preprocessing</span> <span class="kn">import</span> <span class="n">fillMissingValues</span>
<span class="kn">from</span> <span class="nn">preprocessing</span> <span class="kn">import</span> <span class="n">renameDataCategories</span>
<span class="kn">from</span> <span class="nn">preprocessing</span> <span class="kn">import</span> <span class="n">expandFare</span>
<span class="kn">from</span> <span class="nn">preprocessing</span> <span class="kn">import</span> <span class="n">createNewFeatures</span>
<span class="kn">from</span> <span class="nn">preprocessing</span> <span class="kn">import</span> <span class="n">prepareTestDataset</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[4]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">train_test_split</span>
<span class="kn">from</span> <span class="nn">sklearn.metrics</span> <span class="kn">import</span> <span class="n">classification_report</span>

<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">KFold</span><span class="p">,</span> <span class="n">StratifiedKFold</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="kn">import</span> <span class="n">GridSearchCV</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[5]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">sklearn.neighbors</span> <span class="kn">import</span> <span class="n">KNeighborsClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.svm</span> <span class="kn">import</span> <span class="n">SVC</span>
<span class="kn">from</span> <span class="nn">sklearn.linear_model</span> <span class="kn">import</span> <span class="n">LogisticRegression</span>
<span class="kn">from</span> <span class="nn">sklearn.tree</span> <span class="kn">import</span> <span class="n">DecisionTreeClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">RandomForestClassifier</span><span class="p">,</span> <span class="n">VotingClassifier</span><span class="p">,</span> <span class="n">StackingClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.ensemble</span> <span class="kn">import</span> <span class="n">GradientBoostingClassifier</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">MultinomialNB</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">GaussianNB</span>
<span class="kn">from</span> <span class="nn">sklearn.naive_bayes</span> <span class="kn">import</span> <span class="n">CategoricalNB</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Importing-the-dataset">Importing the dataset<a class="anchor-link" href="#Importing-the-dataset">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[6]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;Datasets/train.csv&quot;</span><span class="p">)</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[6]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[7]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>From the table above we can see the different data types that pandas assigned each column and that we have some columns with missing values. Before starting to work with these, we have to clean the data.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Renaming-of-the-columns-&amp;-changing-index">Renaming of the columns &amp; changing index<a class="anchor-link" href="#Renaming-of-the-columns-&amp;-changing-index">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we change some of the names of the columns that doesn't sound that meaningful. (For instance SibSp or Parch). Also we make the index the passenger id, as it makes more sense there.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[8]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">renamed_columns</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;Pclass&quot;</span><span class="p">:</span><span class="s2">&quot;Economic status&quot;</span><span class="p">,</span><span class="s2">&quot;SibSp&quot;</span><span class="p">:</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">,</span><span class="s2">&quot;Parch&quot;</span><span class="p">:</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">}</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">renamed_columns</span><span class="p">,</span><span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[9]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">set_index</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;PassengerId&quot;</span><span class="p">],</span><span class="n">inplace</span> <span class="o">=</span> <span class="kc">True</span><span class="p">)</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="s2">&quot;PassengerId&quot;</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Data-types">Data types<a class="anchor-link" href="#Data-types">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now we will convert some columns to more appropriate data type, which will make things easier to work later. Additionally, this reduces the memory usage of the dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>The columns with object data type are candidates to be of categorical type. For this we check the cardinality they have.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[10]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">select_dtypes</span><span class="p">(</span><span class="n">include</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;object&#39;</span><span class="p">])</span><span class="o">.</span><span class="n">nunique</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[10]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Name        891
Sex           2
Ticket      681
Cabin       147
Embarked      3
dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[11]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Sex&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Sex&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">&quot;category&quot;</span><span class="p">)</span>
<span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Embarked&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Embarked&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">&quot;category&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Other column that could be categorical is the one representing the economic status.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[12]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Economic status&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Economic status&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">&quot;category&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Not always is about converting columns to categorical data types, we can also convert numerical types that use 64 bits to smaller sizes (such as 8 bits, 16, 32). By doing this we can reduce even further the memory usage of the dataset.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Before converting SibSp/Parch to a an integer of smaller size, we have to check the maximum number they have. <em>(Max int of int8:127(signed))</em></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[13]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[[</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">,</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">]]</span><span class="o">.</span><span class="n">max</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[13]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Number of siblings/spouses    8
Number of parents/children    6
dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As there is no problem we can make the conversions.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[14]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">&quot;int8&quot;</span><span class="p">)</span>
<span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">&quot;int8&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[15]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">&quot;int8&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Missing-values">Missing values<a class="anchor-link" href="#Missing-values">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This is one of the most important things to do in data analysis. Let's see what we got.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[16]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">sum</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[16]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Survived                        0
Economic status                 0
Name                            0
Sex                             0
Age                           177
Number of siblings/spouses      0
Number of parents/children      0
Ticket                          0
Fare                            0
Cabin                         687
Embarked                        2
dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's start with the embarked values. As these are 2 cases and we can find the information missing online with some research, we can complete this with the real values. If there were more we could use the mode.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[17]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">embarked_is_null</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Embarked&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">embarked_is_null</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[17]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Economic status</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Number of siblings/spouses</th>
      <th>Number of parents/children</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>1</td>
      <td>1</td>
      <td>Icard, Miss. Amelie</td>
      <td>female</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>830</th>
      <td>1</td>
      <td>1</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>female</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>113572</td>
      <td>80.0</td>
      <td>B28</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[18]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="mi">62</span><span class="p">,</span><span class="s2">&quot;Embarked&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;S&quot;</span>
<span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="mi">830</span><span class="p">,</span><span class="s2">&quot;Embarked&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;S&quot;</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For the cabin we have a lot of missing values, so for now we are going to mark them with a '-'.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[19]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Cabin&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="s2">&quot;-&quot;</span><span class="p">,</span><span class="n">inplace</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>For the age we can use the title to predict it, so we split the name and use the mean age for that title.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[20]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Surname&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;, &#39;</span><span class="p">,</span> <span class="n">expand</span><span class="o">=</span><span class="kc">True</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
<span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span> <span class="o">=</span>  <span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Name&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;, &#39;</span><span class="p">,</span> <span class="n">expand</span><span class="o">=</span><span class="kc">True</span><span class="p">)[</span><span class="mi">1</span><span class="p">]</span><span class="o">.</span><span class="n">str</span><span class="o">.</span><span class="n">split</span><span class="p">(</span><span class="s1">&#39;. &#39;</span><span class="p">,</span> <span class="n">expand</span><span class="o">=</span><span class="kc">True</span><span class="p">)[</span><span class="mi">0</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Before we change them we mark these cases so that if we want we can assign them less importance.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[21]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Completed age&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[22]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">title_count</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Title&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">title_count</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[22]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Mr          517
Miss        182
Mrs         125
Master       40
Dr            7
Rev           6
Major         2
Col           2
Mlle          2
Mme           1
th            1
Don           1
Sir           1
Capt          1
Lady          1
Ms            1
Jonkheer      1
Name: Title, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see that we have some uncommon titles, for these we check if any of these is null.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[23]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">uncommon_titles</span> <span class="o">=</span> <span class="n">title_count</span><span class="p">[</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Title&quot;</span><span class="p">]]</span> <span class="o">&lt;</span> <span class="mi">8</span>
<span class="n">uncommon_titles</span><span class="o">.</span><span class="n">index</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">index</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[24]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">null_age</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[25]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">uncommon_titles</span> <span class="o">&amp;</span> <span class="n">null_age</span><span class="p">,[</span><span class="s2">&quot;Title&quot;</span><span class="p">,</span><span class="s2">&quot;Age&quot;</span><span class="p">]]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[25]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Title</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>767</th>
      <td>Dr</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We have only one person with an uncommon title (Dr.), but in this case we have some other cases to predict this one.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[26]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">age_by_title</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Title&quot;</span><span class="p">)[</span><span class="s2">&quot;Age&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">agg</span><span class="p">(</span><span class="s2">&quot;mean&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[27]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">age_for_nan</span> <span class="o">=</span> <span class="n">age_by_title</span><span class="p">[</span><span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">null_age</span><span class="p">,</span><span class="s2">&quot;Title&quot;</span><span class="p">]]</span>
<span class="n">age_for_nan</span><span class="o">.</span><span class="n">index</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="n">null_age</span><span class="p">]</span><span class="o">.</span><span class="n">index</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[28]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">null_age</span><span class="p">,</span><span class="s2">&quot;Age&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">age_for_nan</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now we change these values on the dataset to prevent overfitting later.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[29]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">uncommon_titles</span><span class="p">,</span><span class="s2">&quot;Title&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="s2">&quot;Other&quot;</span>
<span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Title&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="s2">&quot;category&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Finally we check if we missed any missing value.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[30]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">isnull</span><span class="p">()</span><span class="o">.</span><span class="n">any</span><span class="p">()</span><span class="o">.</span><span class="n">any</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[30]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>False</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Feature-Engineering">Feature Engineering<a class="anchor-link" href="#Feature-Engineering">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p><em>Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques.</em></p>
<p>Now that we have explored many of the features of the dataset, we are going to create new features that may prove useful to solve the problem. For example, categorize the age or make a new variable with the sum of the family members.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[31]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">createNewFeatures</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[32]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">head</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[32]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Economic status</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Number of siblings/spouses</th>
      <th>Number of parents/children</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Surname</th>
      <th>Title</th>
      <th>Completed age</th>
      <th>Family size</th>
      <th>Discrete age</th>
      <th>Categorized age</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>-</td>
      <td>S</td>
      <td>Braund</td>
      <td>Mr</td>
      <td>False</td>
      <td>2</td>
      <td>(20, 25]</td>
      <td>Youth</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>Cumings</td>
      <td>Mrs</td>
      <td>False</td>
      <td>2</td>
      <td>(35, 40]</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>-</td>
      <td>S</td>
      <td>Heikkinen</td>
      <td>Miss</td>
      <td>False</td>
      <td>1</td>
      <td>(25, 30]</td>
      <td>Adult</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>As the price of the ticket was given as a group, we can create a new variable which takes into account the individual price.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[33]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">expandFare</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[34]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">tail</span><span class="p">(</span><span class="mi">3</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[34]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Economic status</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Number of siblings/spouses</th>
      <th>Number of parents/children</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Surname</th>
      <th>Title</th>
      <th>Completed age</th>
      <th>Family size</th>
      <th>Discrete age</th>
      <th>Categorized age</th>
      <th>Ticket size</th>
      <th>Individual fare</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>889</th>
      <td>0</td>
      <td>3</td>
      <td>Johnston, Miss. Catherine Helen "Carrie"</td>
      <td>female</td>
      <td>21.773973</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.45</td>
      <td>-</td>
      <td>S</td>
      <td>Johnston</td>
      <td>Miss</td>
      <td>True</td>
      <td>4</td>
      <td>(20, 25]</td>
      <td>Youth</td>
      <td>2</td>
      <td>11.725</td>
    </tr>
    <tr>
      <th>890</th>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.000000</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.00</td>
      <td>C148</td>
      <td>C</td>
      <td>Behr</td>
      <td>Mr</td>
      <td>False</td>
      <td>1</td>
      <td>(25, 30]</td>
      <td>Adult</td>
      <td>1</td>
      <td>30.000</td>
    </tr>
    <tr>
      <th>891</th>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.000000</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.75</td>
      <td>-</td>
      <td>Q</td>
      <td>Dooley</td>
      <td>Mr</td>
      <td>False</td>
      <td>1</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>1</td>
      <td>7.750</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Data-conversions">Data conversions<a class="anchor-link" href="#Data-conversions">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we change some of the texts the data has, so that they are more meaningful.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[35]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">renameDataCategories</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Reordering-columns">Reordering columns<a class="anchor-link" href="#Reordering-columns">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Finally, we order the dataset in a more relevant way.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[36]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">personal_info</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Surname&quot;</span><span class="p">,</span><span class="s2">&quot;Title&quot;</span><span class="p">,</span><span class="s2">&quot;Name&quot;</span><span class="p">,</span><span class="s2">&quot;Sex&quot;</span><span class="p">,</span><span class="s2">&quot;Age&quot;</span><span class="p">,</span><span class="s2">&quot;Completed age&quot;</span><span class="p">,</span><span class="s2">&quot;Discrete age&quot;</span><span class="p">,</span><span class="s2">&quot;Categorized age&quot;</span><span class="p">]</span>
<span class="n">economic_status</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Economic status&quot;</span><span class="p">,</span><span class="s2">&quot;Fare&quot;</span><span class="p">,</span><span class="s2">&quot;Individual fare&quot;</span><span class="p">]</span>
<span class="n">family</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">,</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">,</span><span class="s2">&quot;Family size&quot;</span><span class="p">]</span>
<span class="n">journey</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Cabin&quot;</span><span class="p">,</span><span class="s2">&quot;Embarked&quot;</span><span class="p">,</span><span class="s2">&quot;Ticket&quot;</span><span class="p">,</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span>
<span class="n">new_order</span> <span class="o">=</span> <span class="n">personal_info</span> <span class="o">+</span> <span class="n">economic_status</span> <span class="o">+</span> <span class="n">family</span> <span class="o">+</span> <span class="n">journey</span>
<span class="n">titanic_df</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">reindex</span><span class="p">(</span><span class="n">columns</span> <span class="o">=</span> <span class="n">new_order</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Dataset-after-handling-it">Dataset after handling it<a class="anchor-link" href="#Dataset-after-handling-it">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[37]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
Int64Index: 891 entries, 1 to 891
Data columns (total 18 columns):
 #   Column                      Non-Null Count  Dtype   
---  ------                      --------------  -----   
 0   Surname                     891 non-null    object  
 1   Title                       891 non-null    category
 2   Name                        891 non-null    object  
 3   Sex                         891 non-null    category
 4   Age                         891 non-null    float64 
 5   Completed age               891 non-null    bool    
 6   Discrete age                891 non-null    category
 7   Categorized age             891 non-null    category
 8   Economic status             891 non-null    category
 9   Fare                        891 non-null    float64 
 10  Individual fare             891 non-null    float64 
 11  Number of siblings/spouses  891 non-null    int8    
 12  Number of parents/children  891 non-null    int8    
 13  Family size                 891 non-null    int8    
 14  Cabin                       891 non-null    object  
 15  Embarked                    891 non-null    category
 16  Ticket                      891 non-null    object  
 17  Survived                    891 non-null    int8    
dtypes: bool(1), category(6), float64(3), int8(4), object(4)
memory usage: 106.8+ KB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[38]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[38]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Surname</th>
      <th>Title</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Completed age</th>
      <th>Discrete age</th>
      <th>Categorized age</th>
      <th>Economic status</th>
      <th>Fare</th>
      <th>Individual fare</th>
      <th>Number of siblings/spouses</th>
      <th>Number of parents/children</th>
      <th>Family size</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Ticket</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Braund</td>
      <td>Mr</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>Male</td>
      <td>22.0</td>
      <td>False</td>
      <td>(20, 25]</td>
      <td>Youth</td>
      <td>Lower</td>
      <td>7.2500</td>
      <td>7.2500</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>A/5 21171</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cumings</td>
      <td>Mrs</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>Female</td>
      <td>38.0</td>
      <td>False</td>
      <td>(35, 40]</td>
      <td>Adult</td>
      <td>Upper</td>
      <td>71.2833</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>C85</td>
      <td>Cherbourg</td>
      <td>PC 17599</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Heikkinen</td>
      <td>Miss</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>Female</td>
      <td>26.0</td>
      <td>False</td>
      <td>(25, 30]</td>
      <td>Adult</td>
      <td>Lower</td>
      <td>7.9250</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>STON/O2. 3101282</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Futrelle</td>
      <td>Mrs</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>Female</td>
      <td>35.0</td>
      <td>False</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Upper</td>
      <td>53.1000</td>
      <td>26.5500</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>C123</td>
      <td>Southhampton</td>
      <td>113803</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Allen</td>
      <td>Mr</td>
      <td>Allen, Mr. William Henry</td>
      <td>Male</td>
      <td>35.0</td>
      <td>False</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Lower</td>
      <td>8.0500</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>373450</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Exploratory-Data-Analysis">Exploratory Data Analysis<a class="anchor-link" href="#Exploratory-Data-Analysis">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>First, let's start by asking some simple questions that will get us closer to the question that matters. What sorts of people were more likely to survive?</p>
<ul>
<li>How many survived?</li>
<li>How much does the sex determine the chances of survival?</li>
<li>What about the age?</li>
<li>Does the economic status helps to determine it?</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="How-many-survived-the-disaster?">How many survived the disaster?<a class="anchor-link" href="#How-many-survived-the-disaster?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[39]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">survival_values</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Survived&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">names</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Died&#39;</span><span class="p">,</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
 
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">survival_values</span><span class="o">.</span><span class="n">index</span><span class="p">,</span><span class="n">height</span><span class="o">=</span><span class="n">survival_values</span><span class="o">.</span><span class="n">values</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;lightcoral&#39;</span><span class="p">,</span> <span class="s1">&#39;lightgreen&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">survival_values</span><span class="o">.</span><span class="n">index</span><span class="p">,</span><span class="n">names</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Amount of people that survived&quot;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">pie</span><span class="p">(</span><span class="n">survival_values</span><span class="p">,</span> <span class="n">labels</span><span class="o">=</span><span class="n">names</span><span class="p">,</span><span class="n">colors</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;lightcoral&#39;</span><span class="p">,</span> <span class="s1">&#39;lightgreen&#39;</span><span class="p">],</span> <span class="n">autopct</span><span class="o">=</span><span class="s1">&#39;</span><span class="si">%1.0f%%</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Proportion of people that survived&quot;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s1">&#39;Survival numbers&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see that approximately 40% of the passengers survived.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="How-much-does-the-sex-affect-the-chances-of-survival?">How much does the sex affect the chances of survival?<a class="anchor-link" href="#How-much-does-the-sex-affect-the-chances-of-survival?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's begin by seeing the proportions of the passengers.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[40]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sex_proportions</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Sex&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
<span class="n">circle</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">Circle</span><span class="p">(</span> <span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">),</span> <span class="mf">0.7</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;white&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">dpi</span><span class="o">=</span><span class="mi">80</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">pie</span><span class="p">(</span><span class="n">sex_proportions</span><span class="o">.</span><span class="n">values</span><span class="p">,</span> <span class="n">labels</span><span class="o">=</span><span class="n">sex_proportions</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">colors</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;goldenrod&#39;</span><span class="p">,</span><span class="s1">&#39;salmon&#39;</span><span class="p">],</span><span class="n">autopct</span><span class="o">=</span><span class="s1">&#39;</span><span class="si">%1.0f%%</span><span class="s1">&#39;</span><span class="p">)</span>
<span class="n">p</span><span class="o">=</span><span class="n">plt</span><span class="o">.</span><span class="n">gcf</span><span class="p">()</span>
<span class="n">p</span><span class="o">.</span><span class="n">gca</span><span class="p">()</span><span class="o">.</span><span class="n">add_artist</span><span class="p">(</span><span class="n">circle</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">suptitle</span><span class="p">(</span><span class="s1">&#39;Proportion of passengers&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now let's see the survival rate.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[41]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">survival_by_sex</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Sex&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">agg</span><span class="p">(</span><span class="s2">&quot;mean&quot;</span><span class="p">)</span>
<span class="n">survival_by_sex</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[41]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Sex
Female    0.742038
Male      0.188908
Name: Survived, dtype: float64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[42]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">4</span><span class="p">,</span> <span class="mi">4</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">survival_by_sex</span><span class="o">.</span><span class="n">index</span><span class="p">,</span><span class="n">height</span><span class="o">=</span><span class="n">survival_by_sex</span><span class="o">.</span><span class="n">values</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;palevioletred&#39;</span><span class="p">,</span> <span class="s1">&#39;cadetblue&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">(</span><span class="n">top</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Proportions of people that survived by sex&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[42]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Text(0.5, 1.0, &#39;Proportions of people that survived by sex&#39;)</pre>
</div>

</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can clearly see that women proportionally had a greater survival rate than men.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="What-about-the-age?">What about the age?<a class="anchor-link" href="#What-about-the-age?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Can we see a pattern by exploring the age? Let's try to see if the children were more likely to survive.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[43]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">distplot</span><span class="p">(</span> <span class="n">a</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Age&quot;</span><span class="p">],</span> <span class="n">hist</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">kde</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">rug</span><span class="o">=</span><span class="kc">False</span> <span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Age distribution&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see that the main group of people that was on the ship were adults, followed by children and then by old people.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[44]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">15</span><span class="p">,</span> <span class="mi">5</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
<span class="n">g</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">barplot</span><span class="p">(</span><span class="s1">&#39;Discrete age&#39;</span><span class="p">,</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">)</span>
<span class="n">g</span><span class="o">.</span><span class="n">tick_params</span><span class="p">(</span><span class="s1">&#39;x&#39;</span><span class="p">,</span><span class="n">labelrotation</span><span class="o">=</span><span class="mi">35</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Survival by grouped ages&quot;</span><span class="p">)</span>
    
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">sns</span><span class="o">.</span><span class="n">violinplot</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">],</span> <span class="n">y</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Age&quot;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Age distribution and survival&quot;</span><span class="p">)</span>
 
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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+bj00ku57rrraNy4scMBpaZp3749U6ZM4e2332b69Olszcrjf75JpX+TAq5unU+8R0M8K9u2LDd/25jED9nRsk67du24//77adu2rcPJqjcV0UREREREBIDU1FRuu+02rr/+ej788ENmzZpFeno6/l3L8e1ZRbBeewobdsD61OFQIcKFeDM24ktfiyuYB0Q7gQYPHszVV1+tBQPEUW63m6uvvpp+/frxwgsvsGDBAubujGdZhp8bT8ulV4NCTadYCXKChte2JrBgtx+LITExkeHDh3PFFVfgdmsZ1YqmIppIFedLNIAtcV1ERETk1MTHxzNkyBCuuOIKPv30U1599VW+//57fOlr8GasI1SnDYUNOxGJr+V01GrBFObhTV+LL3MDJhwEonMaXXfddQwaNIiEhASHE4r8pF69eowfP56vv/6ayZMns3PnTqasS2bB7iC3tMulSaKGeFaEiIUv9viZtTWB7GB0vsqBAwdyzz33aOGASqQimkgV1/4qvYxFRESkYng8HgYOHMiAAQNYtmwZr776KqtWrcK7bzPefZsJ1moeLaYlN3A6apVkCg7j27sa774tGBsBoHnz5gwdOpT+/fvj9XodTijyy3r27MmMGTP497//zSuvvML6Q/DwV6lc0ryAwS3z8KkpqtzszHHz142JbD4c/ZnQsmVLfvvb39KlSxeHk9U8evctIiIiIiLHZYyhV69e9OrVi7Vr1/Lqq6+yePFivId24D20g1BSAwobdyGc0gSN5zoxV95+fLtX4Tn4A8XfrY4dOzJ06FB69+6Ny6VVUaVq8Pl83HLLLQwYMIDnnnuOL7/8kve2x/NVuo/b2ufSsXbQ6YhVWmEY5vwQzwc74glbQ1xcHLfddhvXXHMNHo/KOU7Qd11EROQkfThr30nvm58b+dn10jz2kuvrliqXiEhl6tChA3/84x/Zvn07s2bNYu7cuZCTjmfTXMKJ9Qg0PotwalMV047BlbsP3+5v8R7acWRb7969GTp0KJ07d3YwmcipadSoEY8//jhffPEFkydPJiMzkz99m0LvBgFuOj2XFJ8WHiitNQe8/G1jIhn50Za+8847j9/85jfUr1/f4WQ1m4poIiIiIiJSai1atOCBBx7g9ttvZ9asWbzzzjsU5maSsHke4YQ6BBp3JVyrmYppgCsnA//uVXgO/whEO/suvPBChg0bRuvW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FlsAUFib3oDdtiYcq7GxkX79+qV2XBHpuOeffx6A7QasyzmS9msowtax9dyMGTNyjkbSpCSaiIiIiIh0GUOHDgU2JL7SUErIDR06VN2uRGqAu69Poo3p3/WSaLAh7tLzkO5BSTQREREREekySkm0wtrlqR3T1izf6Ngikq+5c+eyZMkSiuZdbjy0ktH9lETrjpREExERERGRLmPkyJEA2KolqR2zEI9VOraI5GvZsmUA9K1z6rto1mJQYzOw4blI99BFL0cREREREemJtt56a2BD4isNhVWLNzq2iORr7dq1ABQLnnMkHVeMPcNLz0W6ByXRRERERESky1ifRFu9BJqbUzlmYdWijY4tIvlaty50hSx24SEKS7E3NXXN7qhSWV3eAYiIiKStV78hFX8XEZGub/jw4fTt25fly5dTWLmA5r6dHMeseR2FlSGJtv3226cQoYh0Vl1dSFWsbuq6WbTVzSH2YrGYcySSJiXRRESk29n3mAvyDkFERDJSKBQYN24cDz/8MMXlczudRCusWIB5M4MGDWLEiBEpRSkinTFmzBgKhQJL18Ki1cagxq7XrXPW0pA822677XKORNKk7pwiIiIiItKljBs3DoDisrmdPlbpGOPGjcOs67Z6EelOevXqtb579cvLumbbn1kx7h122CHnSCRNSqKJiIiIiEiXsttuuwFQXDIHvHMtVOqWzN7omCJSG0rJp+cXd70kWrPDi0uURPbFZywAACAASURBVOuOlEQTEREREZEuZc8996ShoYHC2uXrJwXokOYmikvnALDvvvumFJ2IpGG//fYD4K9zGmnuYr05H59fz6I1Bfr377++5ax0D0qiiYiIiIhIl9LY2Mj48eMBKC56tcPHKS57A2tex5AhQ9RaRKTGTJ48mQEDBjB/dZHH5tfnHU673PtaLwCmTZtGY2NjztFImpRE64KG9G5gaJ9QhvRuyDscEREREZGqe+tb3wpA/aKXK29QqGPp3ieydO8ToVC5O1jdwpeA0OJF46GJ1JbGxkamTp0KwL2v9so5mrabu7LA4zHp9573vCfnaCRtXa9zsXDBlJ3zDkFEREREJFeTJ0/mqquuorjsTWz1Mryx38YbmEFxM61XvJm6BS8BMGXKlMziFJGOO+KII7j11lt5fEED/11Qx65D1uUdUqtueaEPjrHPPvusnxxBug+1RBMRERGR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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we can see that children were more likely to survive in both graphs.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[45]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">children</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Age&quot;</span><span class="p">]</span> <span class="o">&lt;</span> <span class="mi">13</span>
<span class="n">survived</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">1</span>
<span class="n">survived_age</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">survived</span> <span class="o">&amp;</span> <span class="n">children</span><span class="p">,</span><span class="s2">&quot;Age&quot;</span><span class="p">]</span>
<span class="n">died_age</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="o">~</span><span class="n">survived</span><span class="p">)</span> <span class="o">&amp;</span> <span class="n">children</span><span class="p">,</span><span class="s2">&quot;Age&quot;</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[46]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">survived_age</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[46]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>42</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[47]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">died_age</span><span class="o">.</span><span class="n">count</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[47]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>31</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Does-the-economic-status-helps-to-determine-it?">Does the economic status helps to determine it?<a class="anchor-link" href="#Does-the-economic-status-helps-to-determine-it?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[48]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">total_class_members</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Economic status&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[49]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">survival_by_status</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Economic status&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
<span class="n">survival_by_status</span> <span class="o">=</span> <span class="n">survival_by_status</span><span class="o">.</span><span class="n">unstack</span><span class="p">()</span><span class="o">.</span><span class="n">reset_index</span><span class="p">()</span>
<span class="n">survival_by_status</span><span class="o">.</span><span class="n">index</span> <span class="o">=</span> <span class="n">survival_by_status</span><span class="p">[</span><span class="s2">&quot;Economic status&quot;</span><span class="p">]</span>
<span class="n">survival_by_status</span> <span class="o">=</span> <span class="n">survival_by_status</span><span class="o">.</span><span class="n">drop</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="s2">&quot;Economic status&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[50]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">renamed_columns</span> <span class="o">=</span> <span class="p">{</span><span class="mi">0</span><span class="p">:</span><span class="s2">&quot;Died&quot;</span><span class="p">,</span><span class="mi">1</span><span class="p">:</span><span class="s2">&quot;Survived&quot;</span><span class="p">}</span>
<span class="n">survival_by_status</span> <span class="o">=</span> <span class="n">survival_by_status</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="n">renamed_columns</span><span class="p">)</span>
<span class="n">survival_by_status</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">name</span> <span class="o">=</span> <span class="kc">None</span>
<span class="n">survival_by_status</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[50]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Died</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Economic status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Upper</th>
      <td>0.37</td>
      <td>0.63</td>
    </tr>
    <tr>
      <th>Middle</th>
      <td>0.53</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>Lower</th>
      <td>0.76</td>
      <td>0.24</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[51]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
 
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
<span class="n">squarify</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">sizes</span><span class="o">=</span><span class="n">total_class_members</span><span class="p">,</span><span class="n">value</span><span class="o">=</span><span class="n">total_class_members</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">total_class_members</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=.</span><span class="mi">8</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;sienna&#39;</span><span class="p">,</span> <span class="s1">&#39;gold&#39;</span><span class="p">,</span><span class="s1">&#39;lawngreen&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>  
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">% o</span><span class="s2">f people according to economic status&quot;</span><span class="p">)</span>   
    
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">x</span><span class="o">=</span><span class="n">survival_by_status</span><span class="o">.</span><span class="n">index</span><span class="p">,</span><span class="n">height</span><span class="o">=</span><span class="n">survival_by_status</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">values</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;gold&#39;</span><span class="p">,</span> <span class="s1">&#39;lawngreen&#39;</span><span class="p">,</span><span class="s1">&#39;sienna&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">(</span><span class="n">top</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">% o</span><span class="s2">f people that survived by class&quot;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>On the graphs we can see that people from lower class were more likely to die than from upper class.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="What-happened-to-the-families?">What happened to the families?<a class="anchor-link" href="#What-happened-to-the-families?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we will see if people with families had more chances of survival.</p>
<p>The dataset includes 2 attributes which count the number of members of each group.</p>
<ul>
<li><p>Number of siblings/spouses (sibsp):</p>
<ul>
<li>Sibling = brother, sister, stepbrother, stepsister</li>
<li>Spouse = husband, wife (mistresses and fiancés were ignored)</li>
</ul>
</li>
<li><p>Number of parents/children (parch):</p>
<ul>
<li>Parent = mother, father</li>
<li>Child = daughter, son, stepdaughter, stepson</li>
</ul>
</li>
</ul>
<p><em>Note: Some children travelled only with a nanny, therefore parch=0 for them.</em></p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[52]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[52]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0    608
1    209
2     28
4     18
3     16
8      7
5      5
Name: Number of siblings/spouses, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[53]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[53]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>0    678
1    118
2     80
5      5
3      5
4      4
6      1
Name: Number of parents/children, dtype: int64</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see that the majority of the passengers traveled alone.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[54]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">new_name</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">:</span><span class="s2">&quot;Total&quot;</span><span class="p">}</span>
<span class="n">sibsp</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Number of siblings/spouses&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">to_frame</span><span class="p">()</span><span class="o">.</span><span class="n">unstack</span><span class="p">()</span>
<span class="n">sibsp</span> <span class="o">=</span> <span class="n">sibsp</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
<span class="n">sibsp</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="n">sibsp</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">get_level_values</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
<span class="n">sibsp</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">name</span> <span class="o">=</span> <span class="kc">None</span>
<span class="n">sibsp</span> <span class="o">=</span> <span class="n">sibsp</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">{</span><span class="mi">0</span><span class="p">:</span><span class="s2">&quot;Died&quot;</span><span class="p">,</span><span class="mi">1</span><span class="p">:</span><span class="s2">&quot;Survived&quot;</span><span class="p">})</span> 
<span class="n">sibsp</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[54]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Died</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Number of siblings/spouses</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.654605</td>
      <td>0.345395</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.464115</td>
      <td>0.535885</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.535714</td>
      <td>0.464286</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.750000</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.833333</td>
      <td>0.166667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[&nbsp;]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[55]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">bar_width</span> <span class="o">=</span> <span class="mf">0.3</span>
<span class="n">start_first_bar</span> <span class="o">=</span> <span class="n">sibsp</span><span class="o">.</span><span class="n">index</span><span class="o">-</span><span class="n">bar_width</span><span class="o">/</span><span class="mi">2</span>
<span class="n">start_second_bar</span> <span class="o">=</span> <span class="n">sibsp</span><span class="o">.</span><span class="n">index</span><span class="o">+</span><span class="n">bar_width</span><span class="o">/</span><span class="mi">2</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">4</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">start_first_bar</span><span class="p">,</span> <span class="n">sibsp</span><span class="p">[</span><span class="s2">&quot;Died&quot;</span><span class="p">],</span> <span class="n">width</span> <span class="o">=</span> <span class="n">bar_width</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s1">&#39;lightcoral&#39;</span><span class="p">,</span> <span class="n">edgecolor</span> <span class="o">=</span> <span class="s1">&#39;black&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Died&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">start_second_bar</span><span class="p">,</span> <span class="n">sibsp</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">],</span> <span class="n">width</span> <span class="o">=</span> <span class="n">bar_width</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s1">&#39;lightgreen&#39;</span><span class="p">,</span> <span class="n">edgecolor</span> <span class="o">=</span> <span class="s1">&#39;black&#39;</span><span class="p">,</span>  <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">sibsp</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;%&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Survival by number of siblings/spouses&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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f//33HXXXTQ3N7Pnnnty8MEHc/HFF/Pud7+7ZrUeccQRrFy5khEjRtRsH5IkSVKRDE7tUCqVWL9uPTOumcGo8aNqvr/mZc3cdNZNlEqlDgWnadOm8frrrzN//nze/OY309zczD333MPLL7/cqTpSSmzZsoW6up2/TAYNGsTo0aM7tQ9JkiSpNzA4dcCo8aMYe/DYosto1erVq3nwwQdZvHgxRx99NABvetObeOc73wnAM888w7hx4/jtb3/LpEmTtq2z5557ct9993HMMcdsu+Tu7rvv5ktf+hKPP/443/nOdzj77LNpbGxkwoQJ2/Z3+eWXc+WVV7J8+XJ++ctfbrtULyIYPXo0Cxcu5IQTTtjWf8GCBXz84x+nubmZoUOH8sILL3D++eezaNEiBgwYwJFHHskVV1zBAQccAMCWLVv4/Oc/z9y5cxk4cCCnn346KaVuOpuSJEnS9pwcoo8YOnQoQ4cO5Y477mDjxo27tK0LLriASy65hMbGRj760Y8yefJkbr755u363HLLLZx66qlExHbtI0aM4P3vf3+r/U866SSGDh3KunXrOPbYYxk6dCj3338/Dz74IEOHDuWEE05g06ZNAFx22WXMnTuX66+/ngcffJCXX36ZhQsX7tJxSZIkSZ1lcOoj6urqmDdvHvPnz2ePPfbgL/7iL/jiF7/I448/3uFtfeUrX2HKlCm85S1vYe+992b69Onccsst25YvW7aMJUuWMGPGjFbXnz59OnfccQfr1q0DYO3atfz0pz/d1v+2225jwIABXHfddRx00EFMnDiRG264gaamJhYvXgzAnDlzmD17NtOmTWPixIlcffXV3kMlSZKkwhic+pBp06axYsUK7rzzTt73vvexePFiDj30UObNm9eh7Rx22GHbPT/llFN49tln+fWvfw3AzTffzKRJkzjwwANbXf/9738/dXV13HnnnQD86Ec/YtiwYbz3ve8FYMmSJTz11FMMGzZs20jZXnvtxYYNG3j66adZs2YNK1eu3G5Ci7q6uh3qkiRJkrqLwamPecMb3sCUKVO46KKLeOihh5g5cyZf/vKXGTAg+1VX3if0+uuvt7qNN77xjds9HzNmDMcee+y2Uadbb721zdEmyCaL+OhHP7qt/y233MLJJ5+8bZKJrVu3MnnyZB577LHtHsuWLePUU0/t/MFLkiRJNWJw6uMOPPBA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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In the graph we can see that those with one or two siblings/spouses seem more likely to survive. Let's see now with the parents and children.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[56]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">new_name</span> <span class="o">=</span> <span class="p">{</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">:</span><span class="s2">&quot;Total&quot;</span><span class="p">}</span>
<span class="n">parch</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Number of parents/children&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">to_frame</span><span class="p">()</span><span class="o">.</span><span class="n">unstack</span><span class="p">()</span>
<span class="n">parch</span> <span class="o">=</span> <span class="n">parch</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
<span class="n">parch</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="n">parch</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">get_level_values</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span>
<span class="n">parch</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">name</span> <span class="o">=</span> <span class="kc">None</span>
<span class="n">parch</span> <span class="o">=</span> <span class="n">parch</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">{</span><span class="mi">0</span><span class="p">:</span><span class="s2">&quot;Died&quot;</span><span class="p">,</span><span class="mi">1</span><span class="p">:</span><span class="s2">&quot;Survived&quot;</span><span class="p">})</span> 
<span class="n">parch</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[56]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Died</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Number of parents/children</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.656342</td>
      <td>0.343658</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.449153</td>
      <td>0.550847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.500000</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.400000</td>
      <td>0.600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.800000</td>
      <td>0.200000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[57]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">bar_width</span> <span class="o">=</span> <span class="mf">0.3</span>
<span class="n">start_first_bar</span> <span class="o">=</span> <span class="n">parch</span><span class="o">.</span><span class="n">index</span><span class="o">-</span><span class="n">bar_width</span><span class="o">/</span><span class="mi">2</span>
<span class="n">start_second_bar</span> <span class="o">=</span> <span class="n">parch</span><span class="o">.</span><span class="n">index</span><span class="o">+</span><span class="n">bar_width</span><span class="o">/</span><span class="mi">2</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">4</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">start_first_bar</span><span class="p">,</span> <span class="n">parch</span><span class="p">[</span><span class="s2">&quot;Died&quot;</span><span class="p">],</span> <span class="n">width</span> <span class="o">=</span> <span class="n">bar_width</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s1">&#39;lightcoral&#39;</span><span class="p">,</span> <span class="n">edgecolor</span> <span class="o">=</span> <span class="s1">&#39;black&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Died&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">start_second_bar</span><span class="p">,</span> <span class="n">parch</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">],</span> <span class="n">width</span> <span class="o">=</span> <span class="n">bar_width</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="s1">&#39;lightgreen&#39;</span><span class="p">,</span> <span class="n">edgecolor</span> <span class="o">=</span> <span class="s1">&#39;black&#39;</span><span class="p">,</span>  <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">parch</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;%&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Survival by number of parents/children&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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q/XAAw9kyZIljBgxotv6kCRJkspkcOqESqVC28o2Zlwxg1HjRnV7f60LW7nuU9dRqVS6FJymTZvGa6+9xjXXXMMee+xBa2srv/rVr3jxxRc3q46UEmvXrqWhYdNvk0GDBjF69OjN6kOSJEnqCwxOXTBq3CjG7j227DLqWrZsGffddx/z5s3jkEMOAeBNb3oTb3/72wF46qmn2H333fn973/PxIkT1++z4447ctddd3HooYeuH3J3++2384//+I88+uijfPOb3+TTn/40zc3NjB8/fn1/l1xyCZdddhmLFi3i7rvvXj9ULyIYPXo0c+fO5aijjlq//U033cRHPvIRWltbGTZsGM899xxnnHEGd9xxBwMGDOCggw7i0ksvZbfddgNg7dq1fOELX2D27NkMHDiQk046iZRSD72akiRJ0oacHGIbMWzYMIYNG8bNN9/M6tWrt+hYX/ziF7nwwgtpbm7m2GOPZdKkSVx//fUbbHPDDTdwwgknEBEbtI8YMYL3vve9dbefMmUKw4YNY+XKlRx22GEMGzaMe+65h/vuu49hw4Zx1FFH8eqrrwJw8cUXM3v2bK666iruu+8+XnzxRebOnbtFz0uSJEnaXAanbURDQwNz5szhmmuuYYcdduCd73wnZ511Fo8++miXj/WVr3yFyZMn8zd/8zfstNNOTJ8+nRtuuGH9+oULFzJ//nxmzJhRd//p06dz8803s3LlSgBWrFjBrbfeun7773//+wwYMIArr7ySvfbaiwkTJnD11VfT0tLCvHnzAJg1axZnnnkm06ZNY8KECVx++eXeQyVJkqTSGJy2IdOmTWPx4sXccsstvPvd72bevHnsu+++zJkzp0vH2W+//TZYPv7443n66af57W9/C8D111/PxIkT2XPPPevu/973vpeGhgZuueUWAH784x8zfPhwjjzySADmz5/PE088wfDhw9dfKXvDG97AqlWrePLJJ1m+fDlLlizZYEKLhoaGjeqSJEmSeorBaRvzute9jsmTJ3POOedw//33M3PmTM4991wGDMhOdfV9Qq+99lrdY7z+9a/fYHnMmDEcdthh6686fe973+vwahNkk0Uce+yx67e/4YYbOO6449ZPMrFu3TomTZrEww8/vMFj4cKFnHDCCZv/5CVJkqRuYnDaxu2555688sor7LzzzgAsWbJk/bqHH36408eZPn06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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In this case we can see again that those with a family were more likely to survive than those that wet alone.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Is-there-any-relationship-with-the-embarkation-port?">Is there any relationship with the embarkation port?<a class="anchor-link" href="#Is-there-any-relationship-with-the-embarkation-port?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In this section we will see if people that embarked in a certain port survived more than from the others.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[58]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">embark_ports</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Embarked&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[59]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">survival_by_ports</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Embarked&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">agg</span><span class="p">(</span><span class="s2">&quot;mean&quot;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[60]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">4</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
<span class="n">squarify</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">sizes</span><span class="o">=</span><span class="n">embark_ports</span><span class="p">,</span><span class="n">value</span><span class="o">=</span><span class="n">embark_ports</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="n">embark_ports</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=.</span><span class="mi">8</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;darkslategrey&#39;</span><span class="p">,</span> <span class="s1">&#39;slategrey&#39;</span><span class="p">,</span><span class="s1">&#39;cornflowerblue&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>  
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;</span><span class="si">% o</span><span class="s2">f people from each port&quot;</span><span class="p">)</span>   

<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">survival_by_ports</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">survival_by_ports</span><span class="o">.</span><span class="n">values</span><span class="p">,</span> <span class="n">color</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;slategrey&#39;</span><span class="p">,</span> <span class="s1">&#39;cornflowerblue&#39;</span><span class="p">,</span><span class="s1">&#39;darkslategrey&#39;</span><span class="p">],</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.6</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">(</span><span class="n">top</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;%&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Survival by port of embarkation&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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ADZoF77eAnCZUqqzh/xWANawJCeAipaWFrhcrmijEBHRIHC73UhISACABKWUu6d87DpFRERGibxyJVHStAEiEwE8AuBX0FpDvgKgEMBjRyj/Tmg3jYc+FUdZXyIiGkQMNIiI6Gg1AOgCEPk29XQAtT2McyeAVUqp3yqltiilPgCwBMCNIpLVwzj3AUgI++Qedc2JiGjQMNAgIqKjopTyA9gI4OyIQWdDuxcjGjuAYERa6L6YiJelHJqOTynlDn0AtA6wykRENAR4MzgRERnhIQDPi8gGAGsA3AwgH3pXKBG5D0COUuo6Pf/bAJ4Qke8C+ADauzceBrBeKVU11JUnIiLjMdAgIqKjppR6WURSAPwCWtCwFcD5SqkDepYsaIFHKP8zIuIE8H0ADwJoBrAMwH8OacWJiGjQMNAgIiJDKKUeBfBoD8Ouj5L2BwB/GORqERHRMOE9GkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDgGGkREREREZDjzcFeAiI5tjvrnh7sKhB8MdwWIiIj6jS0aRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERERERkOAYaRERkCBFZIiIlIuIVkY0icnov+a0icq+IHBARn4jsE5Ebh6q+REQ0uMzDXQEiIhr5ROQKAA8DWAJgFYBbALwnIhOVUmU9jPYKgAwA3wKwF0A6eFwiIjpucIdORERG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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>We can see that those that embarked in Cherbourg had slightly better chances, though we have to take into account that Southhampton was the port from which most of the people embarked.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Is-there-any-relationship-between-age/fare-and-the-survival?">Is there any relationship between age/fare and the survival?<a class="anchor-link" href="#Is-there-any-relationship-between-age/fare-and-the-survival?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now that we have seen each individual variable in its own, let's start to look for correlations between them.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[61]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">lmplot</span><span class="p">(</span> <span class="n">x</span><span class="o">=</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="s1">&#39;Fare&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">fit_reg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">legend</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">height</span><span class="o">=</span><span class="mi">7</span><span class="p">,</span><span class="n">aspect</span><span class="o">=</span><span class="mf">1.5</span><span class="p">,</span><span class="n">markers</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;x&quot;</span><span class="p">,</span><span class="s2">&quot;o&quot;</span><span class="p">],</span><span class="n">palette</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;red&#39;</span><span class="p">,</span><span class="s1">&#39;green&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Age/Fare with survival&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>In this graph we can clearly see that those that paid a higher fare likely survived. Let's see if we can get a better view by making a graph for each sex.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[62]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Age&quot;</span><span class="p">,</span><span class="s2">&quot;Fare&quot;</span><span class="p">,</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span>
<span class="n">is_male</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Sex&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;Male&quot;</span>
<span class="n">fare_limit</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Fare&quot;</span><span class="p">]</span> <span class="o">&lt;</span> <span class="mi">300</span> <span class="c1"># So that we can see better the lower values</span>
<span class="n">age_fare_male</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">is_male</span> <span class="o">&amp;</span> <span class="n">fare_limit</span><span class="p">,</span><span class="n">columns</span><span class="p">]</span>
<span class="n">age_fare_female</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[(</span><span class="o">~</span><span class="n">is_male</span><span class="p">)</span> <span class="o">&amp;</span> <span class="n">fare_limit</span><span class="p">,</span><span class="n">columns</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[63]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">lmplot</span><span class="p">(</span> <span class="n">x</span><span class="o">=</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="s1">&#39;Fare&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">age_fare_male</span><span class="p">,</span><span class="n">fit_reg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">legend</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">height</span><span class="o">=</span><span class="mi">7</span><span class="p">,</span><span class="n">aspect</span><span class="o">=</span><span class="mf">1.5</span><span class="p">,</span><span class="n">markers</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;x&quot;</span><span class="p">,</span><span class="s2">&quot;o&quot;</span><span class="p">],</span><span class="n">palette</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;red&#39;</span><span class="p">,</span><span class="s1">&#39;green&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Age/Fare with survival, male&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[64]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">sns</span><span class="o">.</span><span class="n">lmplot</span><span class="p">(</span> <span class="n">x</span><span class="o">=</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span> <span class="n">y</span><span class="o">=</span><span class="s1">&#39;Fare&#39;</span><span class="p">,</span><span class="n">data</span><span class="o">=</span><span class="n">age_fare_female</span><span class="p">,</span><span class="n">fit_reg</span><span class="o">=</span><span class="kc">False</span><span class="p">,</span> <span class="n">hue</span><span class="o">=</span><span class="s1">&#39;Survived&#39;</span><span class="p">,</span> <span class="n">legend</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span><span class="n">height</span><span class="o">=</span><span class="mi">7</span><span class="p">,</span><span class="n">aspect</span><span class="o">=</span><span class="mf">1.5</span><span class="p">,</span><span class="n">markers</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;x&quot;</span><span class="p">,</span><span class="s2">&quot;o&quot;</span><span class="p">],</span><span class="n">palette</span><span class="o">=</span><span class="p">[</span><span class="s1">&#39;red&#39;</span><span class="p">,</span><span class="s1">&#39;green&#39;</span><span class="p">])</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Age/Fare with survival, female&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>This graph further confirms what we saw before, women more likely survived. In addition, we can see that also with the children.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's try to draw the boundaries of the socio-economic status by looking at different numerical values.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[65]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Economic status&quot;</span><span class="p">)[</span><span class="s2">&quot;Fare&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">agg</span><span class="p">([</span><span class="s2">&quot;min&quot;</span><span class="p">,</span><span class="s2">&quot;mean&quot;</span><span class="p">,</span><span class="s2">&quot;max&quot;</span><span class="p">,</span><span class="s2">&quot;std&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[65]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>max</th>
      <th>std</th>
    </tr>
    <tr>
      <th>Economic status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Upper</th>
      <td>0.0</td>
      <td>84.15</td>
      <td>512.33</td>
      <td>78.38</td>
    </tr>
    <tr>
      <th>Middle</th>
      <td>0.0</td>
      <td>20.66</td>
      <td>73.50</td>
      <td>13.42</td>
    </tr>
    <tr>
      <th>Lower</th>
      <td>0.0</td>
      <td>13.68</td>
      <td>69.55</td>
      <td>11.78</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Seeing that the minimum fare was 0 for all economic status, I decided to take a look at these particular cases.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[66]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Fare&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="mi">0</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[66]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Surname</th>
      <th>Title</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Completed age</th>
      <th>Discrete age</th>
      <th>Categorized age</th>
      <th>Economic status</th>
      <th>Fare</th>
      <th>Individual fare</th>
      <th>Number of siblings/spouses</th>
      <th>Number of parents/children</th>
      <th>Family size</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Ticket</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>180</th>
      <td>Leonard</td>
      <td>Mr</td>
      <td>Leonard, Mr. Lionel</td>
      <td>Male</td>
      <td>36.00000</td>
      <td>False</td>
      <td>(35, 40]</td>
      <td>Adult</td>
      <td>Lower</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>LINE</td>
      <td>0</td>
    </tr>
    <tr>
      <th>264</th>
      <td>Harrison</td>
      <td>Mr</td>
      <td>Harrison, Mr. William</td>
      <td>Male</td>
      <td>40.00000</td>
      <td>False</td>
      <td>(35, 40]</td>
      <td>Adult</td>
      <td>Upper</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>B94</td>
      <td>Southhampton</td>
      <td>112059</td>
      <td>0</td>
    </tr>
    <tr>
      <th>272</th>
      <td>Tornquist</td>
      <td>Mr</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>Male</td>
      <td>25.00000</td>
      <td>False</td>
      <td>(20, 25]</td>
      <td>Adult</td>
      <td>Lower</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>LINE</td>
      <td>1</td>
    </tr>
    <tr>
      <th>278</th>
      <td>Parkes</td>
      <td>Mr</td>
      <td>Parkes, Mr. Francis "Frank"</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Middle</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>239853</td>
      <td>0</td>
    </tr>
    <tr>
      <th>303</th>
      <td>Johnson</td>
      <td>Mr</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>Male</td>
      <td>19.00000</td>
      <td>False</td>
      <td>(15, 20]</td>
      <td>Youth</td>
      <td>Lower</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>LINE</td>
      <td>0</td>
    </tr>
    <tr>
      <th>414</th>
      <td>Cunningham</td>
      <td>Mr</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Middle</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>239853</td>
      <td>0</td>
    </tr>
    <tr>
      <th>467</th>
      <td>Campbell</td>
      <td>Mr</td>
      <td>Campbell, Mr. William</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Middle</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>239853</td>
      <td>0</td>
    </tr>
    <tr>
      <th>482</th>
      <td>Frost</td>
      <td>Mr</td>
      <td>Frost, Mr. Anthony Wood "Archie"</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Middle</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>239854</td>
      <td>0</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Johnson</td>
      <td>Mr</td>
      <td>Johnson, Mr. Alfred</td>
      <td>Male</td>
      <td>49.00000</td>
      <td>False</td>
      <td>(45, 50]</td>
      <td>Adult</td>
      <td>Lower</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>LINE</td>
      <td>0</td>
    </tr>
    <tr>
      <th>634</th>
      <td>Parr</td>
      <td>Mr</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Upper</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>112052</td>
      <td>0</td>
    </tr>
    <tr>
      <th>675</th>
      <td>Watson</td>
      <td>Mr</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Middle</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>239856</td>
      <td>0</td>
    </tr>
    <tr>
      <th>733</th>
      <td>Knight</td>
      <td>Mr</td>
      <td>Knight, Mr. Robert J</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Middle</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>239855</td>
      <td>0</td>
    </tr>
    <tr>
      <th>807</th>
      <td>Andrews</td>
      <td>Mr</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>Male</td>
      <td>39.00000</td>
      <td>False</td>
      <td>(35, 40]</td>
      <td>Adult</td>
      <td>Upper</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A36</td>
      <td>Southhampton</td>
      <td>112050</td>
      <td>0</td>
    </tr>
    <tr>
      <th>816</th>
      <td>Fry</td>
      <td>Mr</td>
      <td>Fry, Mr. Richard</td>
      <td>Male</td>
      <td>32.36809</td>
      <td>True</td>
      <td>(30, 35]</td>
      <td>Adult</td>
      <td>Upper</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>B102</td>
      <td>Southhampton</td>
      <td>112058</td>
      <td>0</td>
    </tr>
    <tr>
      <th>823</th>
      <td>Reuchlin</td>
      <td>Other</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>Male</td>
      <td>38.00000</td>
      <td>False</td>
      <td>(35, 40]</td>
      <td>Adult</td>
      <td>Upper</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>19972</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>By investigating a little, it turned out that these people boarded the ship without paying a fare for different reasons. For example, Andrews, Mr. Thomas Jr. was the naval architect for the ship, while other passengers were part of the <a href="https://www.encyclopedia-titanica.org/titanic-guarantee-group/">"guarantee group"</a>.</p>
<p>Let's continue without taking into account these cases.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[67]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">passengers_with_fare</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Fare&quot;</span><span class="p">]</span><span class="o">&gt;</span><span class="mi">0</span><span class="p">]</span>
<span class="n">fare_range_by_status</span> <span class="o">=</span> <span class="n">passengers_with_fare</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Economic status&quot;</span><span class="p">)[</span><span class="s2">&quot;Fare&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">agg</span><span class="p">([</span><span class="s2">&quot;min&quot;</span><span class="p">,</span><span class="s2">&quot;mean&quot;</span><span class="p">,</span><span class="s2">&quot;max&quot;</span><span class="p">,</span><span class="s2">&quot;std&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
<span class="n">fare_range_by_status</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[67]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>mean</th>
      <th>max</th>
      <th>std</th>
    </tr>
    <tr>
      <th>Economic status</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Upper</th>
      <td>5.00</td>
      <td>86.15</td>
      <td>512.33</td>
      <td>78.21</td>
    </tr>
    <tr>
      <th>Middle</th>
      <td>10.50</td>
      <td>21.36</td>
      <td>73.50</td>
      <td>13.08</td>
    </tr>
    <tr>
      <th>Lower</th>
      <td>4.01</td>
      <td>13.79</td>
      <td>69.55</td>
      <td>11.76</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>By taking into account the mean and standard deviation, we can see that some groups overlap between each other in parts, thus blurring the boundaries. Of course here we have to take into account that the fare in this case includes all of the fares from one family.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[68]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">min_fare_std</span> <span class="o">=</span> <span class="n">fare_range_by_status</span><span class="p">[</span><span class="s2">&quot;mean&quot;</span><span class="p">]</span> <span class="o">-</span> <span class="n">fare_range_by_status</span><span class="p">[</span><span class="s2">&quot;std&quot;</span><span class="p">]</span>
<span class="n">max_fare_std</span> <span class="o">=</span> <span class="n">fare_range_by_status</span><span class="p">[</span><span class="s2">&quot;mean&quot;</span><span class="p">]</span> <span class="o">+</span> <span class="n">fare_range_by_status</span><span class="p">[</span><span class="s2">&quot;std&quot;</span><span class="p">]</span>
<span class="n">colors</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;gold&#39;</span><span class="p">,</span><span class="s1">&#39;lawngreen&#39;</span><span class="p">,</span><span class="s1">&#39;sienna&#39;</span><span class="p">]</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">3</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">120</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">hlines</span><span class="p">(</span><span class="n">y</span><span class="o">=</span><span class="n">fare_range_by_status</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">xmin</span><span class="o">=</span><span class="n">min_fare_std</span><span class="p">,</span> <span class="n">xmax</span><span class="o">=</span><span class="n">max_fare_std</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">min_fare_std</span><span class="p">,</span> <span class="n">fare_range_by_status</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s1">&#39;Min fare (std)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">scatter</span><span class="p">(</span><span class="n">max_fare_std</span><span class="p">,</span> <span class="n">fare_range_by_status</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mi">1</span> <span class="p">,</span> <span class="n">linewidths</span><span class="o">=</span><span class="mi">1</span><span class="p">,</span><span class="n">edgecolor</span><span class="o">=</span><span class="s2">&quot;black&quot;</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="s1">&#39;Max fare (std)&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
 
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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m5tL7969WbJkSbHbyNejRw8GDBgApO4GFUJg0KBBALz44oucdtpp7L777tSqVYs2bdpw4YUXsmLFiiLbGD58OCEE3n77bfr27Uvjxo1p3bo1kLo0YfTo0XTs2JHatWvTuHFj+vbty8cfl34ZxYYNGxg7dixnnXXWFkf377vvPg488EDq1atH/fr1adeuHddcc03B9+OMM84AoGfPngWXTowfP76gpttuu42WLVtSq1YtDj74YJ577rliazjkkEPYb7/9uP/++0uttzIY+CVJkiSpsG8fgbxvKdNNvVZX7lRhDRo0oG/fvowbN66gbeLEiVSrVo1+/fpt07aGDBlCTk4OjzzyCLfddhszZ84sCPMlGT16NNdddx0ADz30EHPmzOH6668HYNGiRRx22GHcd999vPDCCwwbNow33niDww8/nI0bN26xrd69e9OmTRueeOKJgoB84YUX8otf/IJjjjmGp556itGjR7Nw4UK6du3KsmWZN10r6o033mDlypX07NmzSPujjz7KxRdfTPfu3XnyySd56qmnuPzyy1m7NnXH9pNPPpmbb74ZgHvvvZc5c+YwZ84cTj75ZABGjBjB//zP/3Dsscfy1FNPMXToUM4//3w++OCDYuvo0aMHzz33HLEMd0+raJV5Wz5JkiRJqvq+e30b+m71auIKMXjwYHr27MnChQtp374948aN44wzztjm6+hPOOEEfve73xV8vWrVKq666iqWLl1K8+bNi11n//33Lzga36FDBzp16lSw7KKLLir4d4yRrl270qNHD1q2bMlzzz1Hr169imxr4MCBjBgxouDruXPnMmbMGO68806uuOKKgvYjjjiCtm3b8tvf/pZbby1purjUXAUABx98cJH2119/nUaNGhV5rUcf/Z+rzJs2bco+++xT8PoKXwKxevVqbr31Vk4//XT+7//+r6C9ffv2dOvWjX333XeLOg4++GDuu+8+PvjgA9q1a1divZXBI/ySJEmSVMSWR6NLFDdVXBkl6N69O61bt2bcuHHMnz+fefPmlXg6/9ZkBvAf/ehHAHz22Wc/qK7ly5dz0UUXsccee1CjRg1ycnJo2bIlAO+///4W/fv06VPk62nTphFCYMCAAWzatKng0bx5cw488MBi705Q2JIlSwgh0KRJkyLtXbp0YfXq1fTv358pU6ZscYnB1syZM4fvv/+es88+u0h7165dC15bpmbNmgGwePHiMu+noniEX5IkSZIKy2lTxo4Bapa1b/kJIXDuuefyu9/9ju+//562bdtyxBFHbPN2cnNzi3y90047AbBu3bpt3lZeXh7HHXccS5Ys4frrr+eAAw6gbt265OXlceihhxa7zV133bXI18uWLSPGyC677FLsPvbee++t1rBu3TpycnKoXr3oLcp/9rOfsWnTJsaMGUOfPn3Iy8ujc+fOjBw5kmOPPXar21y5MnVjuOLOeCjpLIhatWoV1JNtBn5JkiRJKqzhubDyN5R+DX+ERtm5M9agQYMYNmwY999/PzfddFNWaihswYIFvPvuu4wfP56BAwcWtH/00UclrhNC0TsgNGnShBACr776asGHD4UV15a5/oYNG1i7di1169Ytsuzcc8/l3HPPZe3atbzyyivccMMNnHLKKXz44YclHqmH/3wosnTp0i2WLV26tNhb+K1ataqgnmzzlH5JkiRJKqxmK2hYhlPka+yR+nAgC1q0aMGVV17JqaeeWiRgZ0t+eM8M5Q888ECZt3HKKacQY2Tx4sV06tRpi8cBBxyw1fXzr5dftGhRiX3q1q3LiSeeyLXXXsuGDRsKbjFY0tkNhx56KLVq1WLChAlF2mfPnl3ipQ8ff/wx1apVK/b6/srmEX5JkiRJyrTLvbB5Fax5MmNB+lZ9NXaHPV6A6o2yUR0Ao0aNytq+M7Vr147WrVvz61//mhgjO++8M1OnTuXFF18s8za6devGBRdcwLnnnstbb73FkUceSd26dfnyyy957bXXOOCAAxg6dGiJ6/fo0QNITf6XPx8BwPnnn0/t2rXp1q0bu+66K0uXLuWWW26hYcOGdO7cGUhNQAjw4IMPUr9+fWrVqsVee+1Fbm4uv/rVrxg5ciRDhgzhjDPO4PPPP2f48OElntI/d+5cOnbsSOPGjcv82iuKR/glSZIkKVO1naDFn6HF01D3BAh1IdSEmu2g2f/CXvNhp+zOwF6V5OTkMHXqVNq2bcuFF15I//79Wb58OS+99NI2beeBBx7g97//Pa+88go//elPOfnkkxk2bBhr166lS5cuW113jz324IgjjmDKlClF2o844ggWLFjAZZddxrHHHsvll19O27ZtefXVV2natCkAe+21F3fddRfvvvsuPXr0oHPnzkydOhWAG2+8kVtuuYUXXniBXr16cc8993D//fcXewR/zZo1TJ8+fYtJ/rIlVIV7A26PQgjtgQULFiygffv22S5HkiRJ2uF9/PHHQOmTuym5Jk2aRL9+/fjss89o0aJFpe9/7NixXHbZZXz++efbfIS/LD+/CxcuzD8boUOMcWFp2/QIvyRJkiQpEXr37k3nzp255ZZbKn3fmzZt4tZbb+Xqq6+uEqfzg4FfkiRJkpQQIQTGjBnDbrvtRl5eXqXu+/PPP2fAgAH88pe/rNT9bo2T9kmSJEmSEqNDhw4Fk/BVpr322othw4ZV+n63xiP8kiRJkiQlkIFfkiRJkqQEMvBLkiRJkpRABn5JkiRJkhLIwC9JkiRJUgIZ+CVJkiRJSiADvyRJkiRJCWTglyRJkqStWLZsGVOnTmXy5Mn8/e9/z0oN48ePJ4RACIGZM2dusTzGSJs2bQgh0KNHj0qvD+C6665jzz33pEaNGjRq1CgrNZRk8ODBnHDCCdu0zuzZsxk+fDirV68u8zqtWrVi0KBBBV9Pnz6devXqsXjx4m3ad3kx8EuSJElSMT799FP69+/PHnvsTq9evejTpw8HHnggXbp0Ydq0aVmpqX79+owdO3aL9lmzZrFo0SLq16+fhapgypQp3HTTTZxzzjnMmjWLl156KSt1FOedd97h4YcfZuTIkdu03uzZsxkxYsQ2Bf5MRx99NF26dOGaa675wdv4bxj4JUmSJCnDBx98wI9/3IVHH32UIw7ZxIMjYMLtcP4ZsHDBW5x66qk8+OCDlV5Xv379mDRpEt9++22R9rFjx3LYYYex5557VnpNAAsWLADg0ksvpVu3bnTq1Om/3uZ33333X28DYNSoUXTp0qVcavohLrnkEiZMmMDnn39e6fs28EuSJElSITFGzjijLytXfsWf74bpD8H5Z8JZp8CDN8L70yJtWgaGDh1aEHQrS//+/QGYOHFiQds333zDpEmTGDx4cLHrjBgxgh//+MfsvPPONGjQgIMPPpixY8cSYyzo89prr5GTk8OvfvWrIuvmX0pQ3FkF+Vq1asV1110HwC677EIIgeHDhwPw2GOPcdxxx7HrrrtSu3Zt9ttvP37961+zdu3aItsYNGgQ9erVY/78+Rx33HHUr1+fo48+GoANGzYwcuRI2rVrx0477UTTpk0599xz+eqrr0r9fi1btownn3ySn/3sZ0Xa8/LyGDlyJPvuuy+1a9emUaNG/OhHP+Luu+8GYPjw4Vx55ZUA7LXXXltcTrFx40auuuoqmjdvTp06dTj88MN58803i63h1FNPpV69eowZM6bUesubgV+SJEmSCpkxYwbz5y/g0gHQ57gtl++5Gzx8SyQvL4/Ro0dXam0NGjSgb9++jBs3rqBt4sSJVKtWjX79+hW7zqeffsqFF17I448/zuTJk+nduzc///nP+c1vflPQ5/DDD2fkyJHceeedPP300wAsXLiQSy65hAEDBnDeeeeVWNOTTz5ZsPz5559nzpw5DBkyBIB//vOfnHTSSYwdO5bnn3+eX/ziFzz++OOceuqpW2xnw4YN9OrVi6OOOoopU6YwYsQI8vLyOO200xg1ahRnnXUWzzzzDKNGjeLFF1+kR48erFu3bqvfrxdeeIGNGzfSs2fPIu233XYbw4cPp3///jzzzDM89thjnHfeeQWn7w8ZMoSf//znAEyePJk5c+YwZ84cDj74YADOP/987rjjDs455xymTJlCnz596N27N19//fUWNdSsWZOuXbvyzDPPbLXWChFj9PEDHkB7IC5YsCBKkiRJyr5FixbFRYsW/dfbGTp0aATiP54lxveLf+S9RzygLbFJk9xyqLx0Dz30UATivHnz4owZM2LhLNK5c+c4aNCgGGOM7du3j927dy9xO5s3b44bN26MN954Y8zNzY15eXkFy/Ly8uJJJ50UGzVqFBcsWBD333//2K5du7hmzZpS67vhhhsiEL/66qsS++Tl5cWNGzfGWbNmRSC+++67BcsGDhwYgThu3Lgi60ycODECcdKkSUXa582bF4E4evTordY1dOjQWLt27SKvM8YYTznllNixY8etrnv77bdHIH7yySdF2t9///0IxMsvv7xI+4QJEyIQBw4cuMW2rr322litWrWtfi/L8vO7YMGCCESgfSxDbvUIvyRJkiQVkn+UtvUeJfcJAfbeA1at2vKIbkXr3r07rVu3Zty4ccyfP5958+aVeDo/wMsvv8wxxxxDw4YNqV69Ojk5OQwbNoyVK1eyfPnygn4hBP7whz9Qv359OnXqxCeffMLjjz9O3bp1f3CtH3/8MWeddRbNmzcv2Hf37t0BeP/997fo36dPnyJfT5s2jUaNGnHqqaeyadOmgkfHjh1p3rx5sXcsKGzJkiU0bdqUEEKR9i5duvDuu+9y8cUX85e//GWLORG2ZsaMGQCcffbZRdrPPPNMatSoUew6zZo1Iy8vj6VLl5Z5P+XBwC9JkiRJhey8884AfPSvkvvECB99Brm5O1dSVf8RQuDcc8/lT3/6E/fffz9t27bliCOOKLbvm2++yXHHpa5LGDNmDK+//jrz5s3j2muvBdjilPjc3Fx69erF999/zwknnMABBxzwg+tcs2YNRxxxBG+88QYjR45k5syZzJs3j8mTJxe77zp16tCgQYMibcuWLWP16tXUrFmTnJycIo+lS5eyYsWKrdawbt06atWqtUX71VdfzR133MHcuXM58cQTyc3N5eijj+att94q9XWtXLkSgObNmxdpr1GjBrm5ucWuk19DaZcglDcDvyRJkiQVcsYZZwBw/6Ml93n9bVj4EZx5ZvHXzVe0QYMGsWLFCu6//37OPffcEvs9+uij5OTkMG3aNM4880y6du261dnqX3zxRe677z66dOnCk08+yaRJk35wjS+//DJLlixh3LhxDBkyhCOPPJJOnTqVeOvAzKPwAE2aNCE3N5d58+YV+yhtDoUmTZqwatWqLdpr1KjBFVdcwdtvv82qVauYOHEin3/+Occff3ypdwfID/WZR+s3bdpU8GFApvwamjRpstVtlzcDvyRJkiQV0r17dw488EfcMwEef27L5Z8uhoFXB6pXr87FF19c+QUCLVq04Morr+TUU09l4MCBJfYLIVCjRg2qV69e0LZu3Tr++Mc/btH3yy+/ZMCAAXTv3p3Zs2fTq1cvzjvvPD755JMfVGN+gN9pp52KtD/wwANl3sYpp5zCypUr2bx5M506ddrise+++251/Xbt2rFy5Uq++eabEvs0atSIvn37cskll7Bq1So+/fTTInVnHpXv0aMHABMmTCjS/vjjj7Np06Zi9/Hxxx+Tm5vLLrvsstV6y1vxFxhIkiRJ0g4qhMATT/yZI488gn5XLOOeCdD/JKhbB2a8AY8+G9iwEcaMeYD9998/a3WOGjWq1D4nn3wyv/3tbznrrLO44IILWLlyJXfccccWIXzz5s3079+fEAKPPPII1atXZ/z48XTs2JF+/frx2muvUbNmzW2qr2vXrjRu3JiLLrqIG264gZycHCZMmMC7775b5m389Kc/ZcKECZx00klcdtlldOnShZycHL744gtmzJjBaaedxumnn17i+j169CDGyBtvvFFwaQOkbpXXoUMH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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="How-is-the-survival-distributed-by-dividing-sex-and-the-socio-economic-status?">How is the survival distributed by dividing sex and the socio-economic status?<a class="anchor-link" href="#How-is-the-survival-distributed-by-dividing-sex-and-the-socio-economic-status?">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[69]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">female</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Sex&quot;</span><span class="p">]</span> <span class="o">==</span> <span class="s2">&quot;Female&quot;</span>
<span class="n">female_passengers</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="n">female</span><span class="p">]</span>
<span class="n">survival_female_class</span> <span class="o">=</span> <span class="n">female_passengers</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Economic status&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">unstack</span><span class="p">()</span>
<span class="n">survival_female_class</span> <span class="o">=</span> <span class="n">survival_female_class</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">{</span><span class="mi">0</span><span class="p">:</span><span class="s2">&quot;Died&quot;</span><span class="p">,</span><span class="mi">1</span><span class="p">:</span><span class="s2">&quot;Survived&quot;</span><span class="p">})</span>
<span class="n">survival_female_class</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">name</span> <span class="o">=</span> <span class="kc">None</span>
<span class="n">survival_female_class</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[69]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Died</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Economic status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Upper</th>
      <td>0.03</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>Middle</th>
      <td>0.08</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>Lower</th>
      <td>0.50</td>
      <td>0.50</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[70]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">male_passengers</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">loc</span><span class="p">[</span><span class="o">~</span><span class="n">female</span><span class="p">]</span>
<span class="n">survival_male_class</span> <span class="o">=</span> <span class="n">male_passengers</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Economic status&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span><span class="o">.</span><span class="n">unstack</span><span class="p">()</span>
<span class="n">survival_male_class</span> <span class="o">=</span> <span class="n">survival_male_class</span><span class="o">.</span><span class="n">rename</span><span class="p">(</span><span class="n">columns</span><span class="o">=</span><span class="p">{</span><span class="mi">0</span><span class="p">:</span><span class="s2">&quot;Died&quot;</span><span class="p">,</span><span class="mi">1</span><span class="p">:</span><span class="s2">&quot;Survived&quot;</span><span class="p">})</span>
<span class="n">survival_male_class</span><span class="o">.</span><span class="n">columns</span><span class="o">.</span><span class="n">name</span> <span class="o">=</span> <span class="kc">None</span>
<span class="n">survival_male_class</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[70]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Died</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>Economic status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Upper</th>
      <td>0.63</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>Middle</th>
      <td>0.84</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>Lower</th>
      <td>0.86</td>
      <td>0.14</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[71]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">5</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
 
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
<span class="n">h1</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">survival_female_class</span><span class="p">,</span><span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Female survival divided by class&quot;</span><span class="p">)</span>
    
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">h2</span> <span class="o">=</span> <span class="n">sns</span><span class="o">.</span><span class="n">heatmap</span><span class="p">(</span><span class="n">survival_male_class</span><span class="p">,</span><span class="n">annot</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Male survival divided by class&quot;</span><span class="p">)</span>

<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we can clearly see that women from the middle and upper status have way higher chances of survival, while for those of lower class the chances drop by around 40%.</p>
<p>Regarding the men, we can see that those from upper class have slightly better chances than those in middle and lower classes.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Expansion-on-the-families">Expansion on the families<a class="anchor-link" href="#Expansion-on-the-families">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Let's try to see if we can get anything new by viewing the sum of the number of siblings/spouses and parents/children.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[72]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">family_size</span> <span class="o">=</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Family size&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">value_counts</span><span class="p">(</span><span class="n">normalize</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span><span class="o">.</span><span class="n">round</span><span class="p">(</span><span class="mi">2</span><span class="p">)</span>
<span class="n">colors</span><span class="o">=</span><span class="p">[</span><span class="s2">&quot;tab:blue&quot;</span><span class="p">,</span><span class="s2">&quot;tab:orange&quot;</span><span class="p">,</span><span class="s2">&quot;tab:green&quot;</span><span class="p">,</span><span class="s2">&quot;tab:red&quot;</span><span class="p">,</span><span class="s2">&quot;tab:purple&quot;</span><span class="p">,</span><span class="s2">&quot;tab:brown&quot;</span><span class="p">,</span><span class="s2">&quot;tab:pink&quot;</span><span class="p">,</span><span class="s2">&quot;tab:gray&quot;</span><span class="p">,</span><span class="s2">&quot;tab:olive&quot;</span><span class="p">]</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">10</span><span class="p">,</span> <span class="mi">4</span><span class="p">),</span> <span class="n">dpi</span><span class="o">=</span><span class="mi">100</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">0</span><span class="p">))</span>
<span class="n">squarify</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">sizes</span><span class="o">=</span><span class="n">family_size</span><span class="p">,</span><span class="n">label</span><span class="o">=</span><span class="n">family_size</span><span class="o">.</span><span class="n">index</span><span class="p">,</span><span class="n">color</span><span class="o">=</span><span class="n">colors</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=.</span><span class="mi">8</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">axis</span><span class="p">(</span><span class="s1">&#39;off&#39;</span><span class="p">)</span>  
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Family size&quot;</span><span class="p">)</span>   

<span class="n">plt</span><span class="o">.</span><span class="n">subplot2grid</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">),</span><span class="n">loc</span><span class="o">=</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">bar</span><span class="p">(</span><span class="n">family_size</span><span class="o">.</span><span class="n">index</span><span class="p">,</span> <span class="n">titanic_df</span><span class="o">.</span><span class="n">groupby</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Family size&quot;</span><span class="p">)[</span><span class="s2">&quot;Survived&quot;</span><span class="p">]</span><span class="o">.</span><span class="n">agg</span><span class="p">(</span><span class="s2">&quot;mean&quot;</span><span class="p">),</span> <span class="n">alpha</span> <span class="o">=</span> <span class="mf">0.6</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylim</span><span class="p">(</span><span class="n">top</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">ticks</span><span class="o">=</span><span class="n">family_size</span><span class="o">.</span><span class="n">index</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;%&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s2">&quot;Survival by family size&quot;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAxoAAAFuCAYAAAAGbG6sAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAPYQAAD2EBqD+naQAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3deZhcZZn38e/d2fdAQkgCiYDIKgICgo4soqAgKuC4C0bwBUFHcUGNG6ijkRkHtxF1lEVUFHdEQHFjEVAERSGsAiEhCSEJZE93tvv945zWotKdhPSpqqT7+7muusJ5zvLcpxPq9K+e55yKzESSJEmSqtTW6gIkSZIk9T4GDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcgYNPW0RMSUispvX55pcy4yIuKRmeaeyjilNrOHciMhm9SdJjRIRB0fETyNiZkR0RMS8iLglIv6nhTU1/D02Ii6JiBmbsN11EXFXI2sp+9k2Ir4fEY+X17SfNbrPuv7X+5mX535dE2to+vVc1evf6gK0VXsrcG9d25wm13ACsKTJfdb7JvDLFtcgST0SES8Hfg5cB3wAmAtMAA4EXg+8r0Wl9cX32I9RXN9OAR4Enmhy/1vCz3wu8HyK89dWyqChnrgrM29rZQGZ+ddW9l/W8CjwaKvrkKQe+gDwMPDSzFxT0/79iPhAVZ1ExBCgPTM3aZSij77HPht4MDO/24rOt4SfeWZ2AH9sZQ3qOadOqXIRsWtEXBwRD0TEioiYHRFXRsQ+ddsdUQ6LvjEizouIuRGxrNx2+4gYERH/FxELytfFETG87hhPmTrVRS2Hln28oYt1J5frDtrA/kMj4nMR8XBEtEfEExFxW+3x6oeYNzK17Lqa7SIizoyIOyJiZUQ8GRE/iohdNvYzlqQGGAMsqAsZAGTmutrl8v3s3PrtupjO2vl+eHREXBQR84EVwOvK9hd3cYwzynXPKZfr32N/FhGPRMR6v8NExJ8i4i81y++IiBvKKUjLI+LOiPhARAzYxJ9Jl8pryx/L9+7ZEfGpiOhXrovy+verLvYbHhGLI+Ir3Rx3p/JcXwLsWXPtOKJcf055jk9ExJKI+EtEnBoRUXecGRHxi4g4LiL+WtZ5T0QcV66fUi4vj4hbI+LAuv03OF2tJ+dYs91rynNZHMXvCg9FxEX1P4uomTq1gWtrRsRONdsdGBE/L39O7eXP4LUbqkeNYdBQT/SLiP61r7J9IrAQ+BDwMuAdwBrgTxGxexfH+QwwDphCMTR/BPA94MfAYuANwH8BJ5XbbrLMvBH4a1lDvXcCf87MP2/gEOcDZwBfKs/lJOCHFBfk7lxFMdxb+3pvuW56zXZfB74A/AY4HjgT2Bu4OSK23+CJSVL1bgEOjogvRXGvRo9+Ga9zEbCa4j3034GfAo9TTMGtNwX4S2b+fQPHmgwcWdsYEXsAzwMurml+JnBZ2e9xwIXA2RTvv5trPPB94LvAq4AfAR8FvghQjtR8GTgqIp5Vt+/JwEigu1/CO6cL/RV4iH9dQzrD005l7a8FTgR+Uvb1sS6OtS8wDTiv3HYx8JOI+ATwNuDDwJuAUcAvohhp2iQ9PEci4vnA5eU5vh54OfBJNj7Tpv7aeiQwG3iMcnpZRLwIuAkYDbyd4u/oDuDy8H6P5stMX76e1oviIpDdvPp3sX0/YABwP3B+TfsR5T4/r9v+82X7F+vafwosrGubAVxSs7xTue+ULurdr6btoLLt5I2c653ATzeyzbmU77vdrN8dWAD8DhhYth1S9v/eum13pPi077xW/z378uWrb70oPkC5seb9fBXFL2wfAobXbZvAuV0co/49ufP991tdbPs/5fvdqJq2Pcvt31nT9pT3WIpfRh8Dvlt3vPOADmBMN+fXVu57EsWHX9vUrLsEmLEJP6PryvpeWdf+f8BaYHK5PILi/sEv1G03HfjdJvZz10a26Tyfj5XXmKj7e1gB7FDTtm9Z+xxgaE37q8r2V3T3M6+p6bqa5c0+R4oPFbP2776LbXai7npet74f8DNgKfDcmvZ7KIJZ/7rtryzPva3Z/2/15ZcjGuqJkyl+Yf/nKzPXlKMbH46IuyNiFcUb+irgWRQXkXq/qFu+p/zzqi7at4266VOb4HsUn5zVjmr8BzCf4hOVDbkVOCYiPhvFVK9N/sQHICLGU9xQNxc4ITNXlauOo3gD/U7diNBjwN8oQpgkNU1mLszMQynezz8EXAHsRvGp+J0RMbYHh/9xF20XAUOA19W0vZUiLFy2gTrXAN8BToyIUQDltKWTgCsyc2HnthGxfzmFZiFFEFgNXErxS+pum3kuSzPz53Vtl1H84n9YWeNSipGVKRExrKzlSGAv4H83s18i4siI+E1ELOZf5/NJipA4rm7zOzJzds1y57X1usxc0UX7M55OLT08x86ZBD+IiNdGxA5Pp+/S/1KMhLwmM/9S9r8rsAfFaBN119erKR5u0NXMCjWIQUM9cU9m3lb7KtvPBz5F8UnDK4CDKS5cf6O4qNSrf5rGqo20D346RWZxQ9nXgTdGxOiI2I5i2Pmb5boNeRfFp2THA78HnohifnD9UPF6ImIExRvbAOCYzFxcs3p7IIB5FBeK2tchQE8u6JK02cr38/My8zUUU2E/T/Hpck9uCJ/bRT/TKX7hfCv8Myy8mSIsbOwpSxdRXAteXy6/lOKXyH9Om4qIyRQjNDsA7wY6Q1Tnh05P64OjGvO6aHus/LN2Wu2XKT71f1O5/E6KG6yv2JxOI+J5wLXl4v8D/o3ifD5dttWfz1N+hjUfdFVybS1t1jlm5g0U19X+FMHv0Yi4K7q4n7IrEfFRimlRp2dm7dOxOqcdf471r60XlOu8vjaRT51SI7wZuDQzP1zbWH4atqg1JfFVik/oTqF4M+0PfG1jO2XmcuAc4JzyvoljgM9SDMHu0d1+5dzmH1PMDz40iyd41FpAMaJxKMWnd/U2FoAkqeEyc3U5p/89FE9C6tQBDOpil+7uX+vuxuKLgQsiYk9gF+rCwgbqujsibqUIKV8v/5zDv34Rh+IX2WHAiZn5SGdjROy3seNvRFf30I0v//znaEpm/iMirgHeUf75SuCczFy7mf2+nuIX5uMys72zMSKO38zj9VhPzjEzrwCuiIhBFB+wTQUui4gZmXlLd/uV91l8imLq3kV1qxeUf06juH+lK/dtrDZVx6ChRkjqflGO4vnsOwD/aElBmXMj4ocUN1wPBK7MzJlP8xjzgEsiYl/grIgYWjf8XOtCiulPx2TXNzT+giL47JCZP3g6dUhSI0TEhMxcb+SBf015rf2epBnAc+r2PxLYnKmt51Pcy7ELxY29125ohxoXA1+NiBdSjJ6fX/cLbme4+ef1KCKCYjSgJ0ZExCvrpk+9EVgH3FC37RcpzudbFFOdvtGDfpNiKvI/z7GczntSD45ZhR6dYzmz4PqIWEQxMrU/xYMJ1hMRLyuPf1FmfqKLY90XEQ8A+9Z/2KnWMGioEX5BMWfzXuDvwAEUT/lo9XPQvwj8qfzvrp50sp6I+BPF+fwdeJLignsScEt3ISMizi63+TKwPCIOqVm9JDPvzsybIuL/gIujeKzgDcByik/zXgjcmZlffbonKEk98KuIeJRixPZeiunV+1HcuLuM8qlKpW8Dn4qITwLXU8zLfyfFk402WWYuioifUgSN0cDnsu5RuhvQGVK+RzG6cknd+l9TTAv6XkT8F8Vo9hnANk+nxi4spAg4kykecnIsRXj5av0HWJn564i4G3gR8J3MfLwH/V5F8QTDy8rrxxjg/bR4BHxzzrH8d7Mj8FuK3w1GU0xvW03x76mrfXameOrjQxTXzkPqNvlrGVpOB66J4tG7l1CE120prt/PLacEqkkMGmqEzjeLqRSfbv2F4tF6/9nKojLz1oiYAazMzN9u4m6/oxgKfg8wlOIN61L+NSe2K3uXf/5H+ap1PeWN3pl5ekT8keJN8UyKi/ociqe83LqJ9UlSVf6T4glE76H40GMQxb0VvwGmZeY9Ndv+N8UjTKdQ/LJ7K8W9b5tz/8HFFI8xh/XDQrcyc3EZUt4I3JSZ99etvzciXk1xXj+hCAiXUYSTazajzk6PUdzn8TlgH4p7Hj5DMc22Kz+geIrTZt8EDpCZv4uIU4APUoTB2RSf7j9OMYreSk/3HP9E8Y3z5wHbUUyrvg04srx3pyvPoPidYjeKe2/q7Uzx5LDfl/ezfITiEfLbUPzd313WqSaKzE36Yk5pqxfFlz/9DXhHZl6wse0lSeqpiLiN4lGx3X457NauL5yjNo8jGur1IuKZFJ+EfIbi07lLWlqQJKlXi4iRFDfPH0cxffiE1lZUvb5wjuo5g4b6go9R3DNxD8Xztru7gVuSpCo8l+KR6AuBT2Tmz1pcTyP0hXNUDzl1SpIkSVLl/MI+SVKPRMRhEXFlRMyJiNyU5/pHxOERcXtEtEfEQxHx9mbUKklqHoOGJKmnhlE8aOGdm7Jx+ZjKqymeHLM/xf1TXyqfECRJ6iWcOiVJqkxEJHDChuZrR8R5wCszc8+atq9RfMnW85tQpiSpCbwZXJLUbM9n/W9//hVwakQMyMzVXe0UEYMovtuh1rYU32MgSWquEcCc3MCoxSYHjZ0+dNVtlZQkCYCzFw1pdQlbpPnjb2h1CZvl0MO+3bBjv/jIBw9s2MFbYzwwr65tHsU1aSzFY6i7MpXuvxRNktR8O1J8eWSXHNGQJLVC/Sdg0U17rWkU3+rcaQTw6KxZsxg5cmSVtUmSNmDJkiVMmjQJYOmGtjNoSJKa7TGKUY1a44A1FM/k71JmdgAdncsRRTYZOXKkQUOStkA+dUqS1Gy3AEfVtR0N3Nbd/RmSpK2PQUOS1CMRMTwi9ouI/cqmncvlyeX6aRFxac0uXwOeERHnR8SeEXEKcCrwuSaXLklqIKdOSZJ66kDg9zXLnfdRfAuYAkwAJneuzMyHI+JY4PPAO4A5wLsy88dNqVaS1BQGDUm9ygMPPDD85ptvHv/4448PXbFixYDjjz/+wX333XdR5/o77rhj9F/+8pft5s+fP7S9vb3/KaeccvekSZNWtrLmrV1mXse/bubuav2ULtquB57buKokSa3m1ClJvcqqVavaxo0bt+Koo46a2dX61atXt+24447LDjvssG4fxydJknrOEQ1Jvcree++9ZO+9914CcMUVV6y3/qCDDnoCYMGCBQOvvbb+O+MkSVJVHNGQJEmSVDlHNKQ+bMGSuQN+fPMFOz4w92+j1qxdFdsOH9/xxsPfN2PXCfusaHVtkiRp62bQkPqopSsX9fv8Fe/eY+ft91p62ks/ef+ooWPWzFs0c9CwQSPWtro2SZK09TNoSH3U1bd9a/zIoduuetvR587obNt+9KRVLSxJkiT1IgYNqY+6Z9afRz9r4n5LvnL1h3Z55PF7R4wYMnr1C/Y49vEX7/vaBa2uTZIkbf0MGlIf9eTy+YNufeDX2/3bni+fd8xz3zz3ocemD7vy1osm9+83MA9/9vELW13f5mpvb2+bP3/+oM7lJ598cuCsWbOGDB06dO2YMWNWLVu2rN+TTz45cMmSJQMB5s+fPxhg5MiRq0eNGrWmVXVLktTbGDSkPiozmbjtTite+8J3zQbYZfyzV859csaQm++9erutOWjMnDlz2Pe+973dOpevv/76Sddffz177rnnwte+9rUzpk+fPvqXv/zlTp3rr7zyyl0ADjnkkLkvfelL57SgZEmSeiWDhtRHDR8yavW4UTs+5Ruxx28zuX36zFu3aVVNVdhtt92WnnPOObd3t/7ggw9eePDBB2+1QUqSpK2F36Mh9VGTt9t92fwlcwbXtj2+aPbg0cPGeEO4JEnqMUc0pD7qxc95zbwv/+L9e1zxp2+MP+hZL3nyocfuGnbbP3479sTnn/FIq2vrjdasSb721YUTb7xx+ZglS9YOGDWq3+ojXjR8wWmnbTu3rS1aXZ4kSZUzaEh91LMm7rvi5COnPnjVny/Z4Xd//9HE0cPGdhx30FtnHbr3K59odW290cUXPTH+179ett1Z7xk7Y9ddB66cPr1j2Je/tGCnYcPa1p500jaPt7o+SZKqZtCQ+rADdz1y8YG7Hrm41XX0Bffe2zH8gAOHLHrRi4YvBpg0aeCq3/9+2bYPPNAxrNW1SZLUCN6jIUlNsNdeg5dNv6t95EMPdQwCuPvu9iH339cx/HnPG2rQkyT1So5oSFITnHLqNo8tX76u3+mnzX52BJlJvO51o2cfd9xIp6pJknolg4YkNcHVVy/d5sYbl4856z1jH3rmMwe233dfx5CLLnxy8tix/VYff8IoH7crSep1DBqStihva39xq0vYLPfx7Q2uv+TiJyedcOLIuS9/+cgnAfbYY/DKeY+tGfSjHy0eb9CQJPVG3qMhSU2walW2tcVTH2Pb1ha5LvHZtpKkXskRDUlqgv2fO3jRj360eML24/uv2nXXgSvvuadj6C9+sWT7I140bEGra5MkqREMGpLUBO9//3Yzv/rVJ3b46gULJy9dunbA6NH9Vr3kJcPnn3b6mLmtrk2SpEYwaEhSEwwf3m/d2WdvNwuY1epaJElqBoOGpF7twK+csM+8ZQsH1refuPfR87943EdntqImSZL6AoOGpF7tqrd8454169b+c/mux+4f8raffmS3V+5x5JMtLEuSpF7PoCG1yJHXvaPVJWyZjv+/Sg+3/fCxa2qXz//DxaMnjhjX8aJnPn9ppR1JkqSn8PG2kvqMjjWr4pf337Dtq/Z6yYL6R81KkqRqGTQk9Rk/mX7t6GWrVvQ/af9X+QV5kiQ1mEFDUp/xwzuvGXvwpH0XTxo1YXWra5EkqbczaEjqEx56YtbA2+fcNfL1z3m5X5AnSVITGDQk9Qnf/usVY0cPHrn6FXscuajVtUiS1BcYNCT1emvXrePn9/x2zMv3eNHCAf182J4kSc1g0JDU6137wI0jH1++cOBJ+73KaVOSJDWJH+1J6vWO2f3wJbM+eMPtra5DkqS+xBENSZIkSZUzaEiSJEmqnEFDkiRJUuUMGpIkSZIqZ9CQJEmSVDmDhiRJkqTKGTQkSZIkVc7v0ZC0Rbn84fNaXcLm+fqeDTv0i49s2KElSWoYRzQkSZIkVc6gIYnz5z8+fq/77j3gI3PnTmp1LZIkqXcwaEh93J9XrBj68yVLtttpwMCVra5FkiT1HgYNqQ9bunZt29S5c3b52LjtZwzv17a21fVIkqTew6Ah9WEfeWzu5EOGDVv84hEjlra6FkmS1Lv41Cmpj/rhokXb3N/RMeynO+18d6trkSRJvY9BQ+qDZq5aNeD8+Y9P/uqOk+4f0taWra5HkiT1PgYNqQ+6Y+XKYYvXrev/5pmP7NXZtg6Y3t4+/Ioli8fdsdvut/ePaGGFkiRpa2fQkPqgI4YPX3L55GdMr237yGNzd548cGD76duOmdtXQ8bP77h74g33Pzyhtm3owAFrPnn80X9rVU1bk4g4EzgbmABMB87KzBs3sP2bgA8AzwIWA78E3p+ZC5tQriSpwQwaUh80sl+/dfsMGdJe2za4rW3dqLZ+a+rb+5oxw4e2v/2IQ+7rXG7ro6Hr6YqI1wFfAM4EbgJOB66JiL0yc2YX278QuBR4D3AlsAPwNeCbwAnNqluS1Dg+dUqSarRF5DZDh6zpfI0aMnhNq2vaSrwXuDAzv5mZ92TmWcAs4Ixutj8EmJGZX8rMhzPzD8DXgQObVK8kqcEc0ZAEwA+esdN9G9+q93ty+cpB51zx6+f0a4vcYfSo5cftu8ej248csarVdW3JImIgcADw2bpV1wIv6Ga3m4FPR8SxwDXAOODfgasaVackqbkMGpJUesaY0csmjBqxYvuRIzqWtLf3/+09/5j4ld/dsufZLzv8rhGDB/mFht0bC/QD5tW1zwPGd7VDZt5c3qNxOTCY4nr0c+A/uuskIgYBg2qaRvSgZklSgzl1SpJK+06auOSgnSctmjxm9Mpn7zB+6emHH/wAwM3/eGRsq2vbStQ/Kjm6aCtWROwFfAn4JMVoyMuAnSnu0+jOVIqbxjtfj/awXklSAxk0JKkbgwcMWLfdiOErFixbPmjjW/dpC4C1rD96MY71Rzk6TQVuysz/zsy/Z+avKG4kPyUiJnSzzzRgVM1rxx5XLklqGIOGJHVj9dq1sXDZ8iEjBw9a3epatmSZuQq4HTiqbtVRFPdidGUoxde31Oqcntblo74ysyMzl3S+gKWbWbIkqQm8R0OSSj+87e87PnuH8YvGDBu6akl7+4BfT//HhFVr1vY75JmT/V6HjTsf+HZE3AbcApwGTKacChUR04AdMvPkcvsrgW9ExBnAryi+e+MLwK2ZOafZxUuSqmfQkKTSkpUdA79/6992Wblqdf+hAwesmTh65PIzj3z+PduNGO5TpzYiMy+PiDHAxylCw13AsZn5SLnJBIrg0bn9JRExAngn8D/AIuB3wAebWrgkqWEMGpJUOvXQgx5qdQ1bs8y8ALigm3VTumj7MvDlBpclSWoR79GQJEmSVDmDhiRJkqTKGTQkSZIkVc6gIUmSJKlyBg1JkiRJlTNoSJIkSaqcQUOSJElS5QwakiRJkipn0JAkSZJUOYOGJEmSpMoZNCRJkiRVrn+rC5D6qtdO9X+/rj3S6gK2OO9rdQGSJG0GRzQkSZIkVc6gIUmSJKlyBg1JkiRJlTNoSJIkSaqcQUOSJElS5XzsjSRJapipP7mzYceeduI+DTu2pJ5zREOSJElS5QwakiRJkipn0JAkSZJUOYOGJEmSpMoZNCRJkiRVzqAhSZIkqXIGDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcgYNSZIkSZUzaEiSJEmqnEFDkiRJUuUMGpIkSZIqZ9CQJEmSVDmDhiRJkqTKGTQkSZIkVc6gIUmSJKlyBg1JkiRJlTNoSJIkSaqcQUOSJElS5QwakiRJkipn0JAkSZJUOYOGJEmSpMoZNCRJkiRVrn+rC5Ck+VfN3+7JG57cbvWi1YMABo0btHK7V2w3Z9TzRi1pdW2SJGnzGDQktdyAbQes2v7E7WcPmjioHeCJ654YO+vrs3YduN3Au4fsPKS91fVJkqSnz6AhqeVGP3/04trliSdNnL3o5kXbLb9/+XCDhiRJWyfv0ZC0Rcm1yRPXPbHNulXr2obtPmxZq+vRpouIMyPi4Yhoj4jbI+LQjWw/KCI+HRGPRERHRDwYEac0q15JUmM5oiFpi7DioRVDHv7sw3vk6mxrG9S2dtJpkx4cspOjGVuLiHgd8AXgTOAm4HTgmojYKzNndrPbD4DtgVOBfwDj8LokSb2Gb+iStgiDJw1uf+bHn3n32uVr+y3+0+JtZn9r9k4Dtx94n2Fjq/Fe4MLM/Ga5fFZEvBQ4A5hav3FEvAw4HNglM58om2c0o1BJUnM4dUrSFqFtQFsO3nFwx7Ddh62YePLE2YMmDFq54JoF27e6Lm1cRAwEDgCurVt1LfCCbnZ7JXAb8IGImB0R90fE5yJiyAb6GRQRIztfwIgq6pckNYYjGpK2WLkmo9U1aJOMBfoB8+ra5wHju9lnF+CFQDtwQnmMC4Btge7u05gKnNPTYiVJzeGIhqSWm/PtOTss/fvS4R1zOwaueGjFkDnfnrPDyodWjhj9gtFPbHxvbUGybjm6aOvUVq57U2bemplXU0y/mrKBUY1pwKia1449L1mS1CiOaEhquTVL1vSffdHsndcsXTOg3+B+aweOH7hy8jsnPzDygJF+Yd/WYQGwlvVHL8ax/ihHp7nA7MysfbTxPRThZEfggfodMrMD6OhcjnDAS5K2ZAYNSS03+R2TH2l1Ddp8mbkqIm4HjgJ+WrPqKOCKbna7CXhNRAzPzM7HGO8GrAMebVixkqSmceqUJKkK5wNvi4hTImLPiPg8MBn4GkBETIuIS2u2vwxYCFwcEXtFxGHAfwMXZebKZhcvSaqeIxqSpB7LzMsjYgzwcWACcBdwbGZ2jlZNoAgendsvi4ijgC9TPH1qIcX3any0qYVLkhrGoCFJqkRmXkDx5Kiu1k3pou1eiulVkqReyKAhSVIfMPUndzb0+NNO3Kehx5e09fEeDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyvl4W0mSmsjHzErqKxzRkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcgYNSZIkSZUzaEiSJEmqnEFDkiRJUuUMGpIkSZIqZ9CQJEmSVDm/GbyXWPngbcMX3/rj8avnPzJ03colA8a8/L0PDn/2kYtaXZckSZL6Jkc0eol1q1e2DRy704ptXnTqzFbXImnrExFjI+LlEfHKiJjQ6nokSVs/RzR6iWF7HLpk2B6HLgFYePXnW12OpK1IRLwauBC4HxgA7B4R78jMi1tbmSRpa+aIhiT1MRExvK7pHOB5mfm8zNwfeA3w6eZXJknqTQwaktT33B4Rr6pZXgOMq1neHljV3JIkSb2NU6ckqe95KXBBREwB3gG8G7g8IvpRXBfWAVNaVp0kqVcwaEhSH5OZM4BjI+KNwPXAF4Fdy1c/4N7MbG9dhZKk3sCpU5LUR2XmZcDzgP2B64C2zLzDkCFJqoIjGr3Euo7lbasXzBzUubxm0WMDO2bfM6Rt6Ki1A7aZ6FxrSU8REccAewF/y8xTI+II4LKIuBr4eGaubGmBkqStniMavUT7rOnDHvvO2Xs99p2z9wJYfNNlkx77ztl7Lbrukomtrk3SliUi/gu4BDgI+HpEfCwzr6MY2egA7iiDiCRJm80RjV5i6K7PW/qMD/7i9lbXIWmrcArw0sy8PSK2Bf4IfCozVwEfjYjvAV8HrmllkZKkrZsjGpLU96wAdi7/exLwlHsyMnN6Zr6w6VVJknoVg4Yk9T1TgUsjYg7FU6c+1uJ6JEm9kFOnJKmPyczvRsQvgV2ABzJzUatrkiT1PgYNSeqDMnMhsLDVdUiSei+nTkmSJEmqnEFDkiRJUuUMGpIkSZIq5z0aUovc+dLYmLgAAA9YSURBVPDMVpcgSZLUMI5oSJIkSaqcQUOSJElS5QwakiRJkipn0JAkSZJUOYOGJEmSpMoZNCRJkiRVzqAhSZIkqXIGDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmViIgzI+LhiGiPiNsj4tBN3O/fImJNRNzR6BolSc1j0JAk9VhEvA74AvBpYH/gRuCaiJi8kf1GAZcCv214kZKkpjJoSJKq8F7gwsz8Zmbek5lnAbOAMzay39eBy4BbGl2gJKm5DBqSpB6JiIHAAcC1dauuBV6wgf3eCjwT+MQm9jMoIkZ2voARm1myJKkJDBqSpJ4aC/QD5tW1zwPGd7VDRDwL+Czwpsxcs4n9TAUW17we3axqJUlNYdCQJFUl65ajizYioh/FdKlzMvP+p3H8acComteOm1mnJKkJ+re6AEnSVm8BsJb1Ry/Gsf4oBxRTng4E9o+I/y3b2oCIiDXA0Zn5u/qdMrMD6OhcjogKSpckNYojGpKkHsnMVcDtwFF1q44Cbu5ilyXAPsB+Na+vAfeV//2nhhUrSWoaRzQkSVU4H/h2RNxG8QSp04DJFAGCiJgG7JCZJ2fmOuCu2p0j4nGgPTPvQpLUKxg0JEk9lpmXR8QY4OPABIogcWxmPlJuMoEieEiS+giDhiSpEpl5AXBBN+umbGTfc4FzKy9KktQy3qMhSZIkqXIGDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcgYNSZIkSZUzaEiSJEmqnEFDkiRJUuUMGpIkSZIqZ9CQJEmSVLn+rS5AkiRJm2/qT+5s2LGnnbhPw46t3s8RDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcn5hnySpz/KLziSpcRzRkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcgYNSZIkSZUzaEiSJEmqnEFDkiRJUuUMGpIkSZIqZ9CQJEmSVDmDhiRJkqTKGTQkSZIkVc6gIUmSJKlyBg1JkiRJlTNoSJIkSaqcQUOSJElS5fq3ugBJfdvU37SPv/L+Nds8vGjd4EH9WPfcCf2W/c/Rgx/dd3y/jlbXJkmSNp8jGpJa6g+z1o447YABj18/Zdg9V71p6P1r1xHHfHfFbks60vcnSZK2Yo5oSGqpG9867IHa5e++um3GDucv2/emmWuGHvOsActaVZckSeoZPzGUtEV5YmX2Axg7tG1Nq2uRJEmbz6AhaYuxLpN3XdM+6bkT2pYdtEO/9lbXI0mSNp9TpyRtMd7ys/bJ9y1YN+SGU4bd2+paJElSzxg0JG0R3vKzlZN+/eCa0ddNGXrvM7dpW93qeiRJUs8YNCS11LpMpvysffIv/7Fm9G9PHnrfHmP7rWp1TZIkqecMGpJa6uSftk++4r7V237/34f8Y9TgWDtz8br+ANsOibXDB0a2uj5JkrR5DBqSWuq7d67eDuC4y1buXtv+xZcNmvGugwctbE1VkiSppwwakloqzxl5e6trUDUi4kzgbGACMB04KzNv7GbbE4EzgP2AQeX252bmr5pUriSpwXy8rSSpxyLidcAXgE8D+wM3AtdExORudjkM+DVwLHAA8HvgyojYvwnlSpKawBENSVIV3gtcmJnfLJfPioiXUoxaTK3fODPPqmv6cES8CngF8NeGVipJagpHNCRJPRIRAylGJa6tW3Ut8IJNPEYbMAJ4otrqJEmt4oiGJKmnxgL9gHl17fOA8Zt4jPcBw4AfdLdBRAyiuJ+j04inUaMkqckMGpKkqtQ/jji6aFtPRLwBOBd4VWY+voFNpwLnbHZ1UoNN/cmdDT3+tBP3aejxpao5dUqS1FMLgLWsP3oxjvVHOZ6ivIn8QuC1mfmbjfQzDRhV89pxs6qVJDWFQUOS1COZuQq4HTiqbtVRwM3d7VeOZFwCvDEzr9qEfjoyc0nnC1i6+VVLkhrNqVOSpCqcD3w7Im4DbgFOAyYDXwOIiGnADpl5crn8BuBS4N3AHyOiczRkZWYubnbxkqTqGTQkST2WmZdHxBjg4xRf2HcXcGxmPlJuMoEieHQ6neIa9JXy1elbwJSGFyxJajiDhiSpEpl5AXBBN+um1C0f0YSSJEkt5D0akiRJkipn0JAkSZJUOYOGJEmSpMoZNCRJkiRVzqAhSZIkqXIGDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcv1bXYAkSVKVpv7kzoYef9qJ+zT0+FJv4YiGJEmSpMoZNCRJkiRVzqAhSZIkqXIGDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcgYNSZIkSZUzaEiSJEmqnEFDkiRJUuUMGpIkSZIqZ9CQJEmSVDmDhiRJkqTKGTQkSZIkVc6gIUmSJKlyBg1JkiRJlTNoSJIkSaqcQUOSJElS5QwakiRJkipn0JAkSZJUOYOGJEmSpMoZNCRJkiRVzqAhSZIkqXIGDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKmfQkCRJklQ5g4YkSZKkyhk0JEmSJFXOoCFJkiSpcgYNSZIkSZUzaEiSKhERZ0bEwxHRHhG3R8ShG9n+8HK79oh4KCLe3qxaJUmNZ9CQJPVYRLwO+ALwaWB/4EbgmoiY3M32OwNXl9vtD3wG+FJEvLo5FUuSGs2gIUmqwnuBCzPzm5l5T2aeBcwCzuhm+7cDMzPzrHL7bwIXAe9vUr2SpAbr3+oCJElbt4gYCBwAfLZu1bXAC7rZ7fnl+lq/Ak6NiAGZubqLfgYBg2qaRgAsWbJkc8oGoGPFss3ed2O6q6uRffa1fvvSubaq3578/6Xea1P/XURmNrgUSVJvFhETgdnAv2XmzTXtHwbekpm7d7HP/cAlmfmZmrYXADcBEzNzbhf7nAucU/0ZSJI2046ZObu7lY5oSJKqUv/JVXTRtrHtu2rvNA04v65tW+CJTaqu50YAjwI7Akt7cZ99rd++dK6t6rdV59qXtOrvdc6GNjBoSJJ6agGwFhhf1z4OmNfNPo91s/0aYGFXO2RmB9BR19y0eR0RnTmIpZnZlH5b0Wdf67cvnWur+m3VufYlLfoZb7QfbwaXJPVIZq4CbgeOqlt1FHDz+nsAcEsX2x8N3NbV/RmSpK2PQUOSVIXzgbdFxCkRsWdEfB6YDHwNICKmRcSlNdt/DXhGRJxfbn8KcCrwuaZXLklqCKdOSZJ6LDMvj4gxwMeBCcBdwLGZ+Ui5yQSK4NG5/cMRcSzweeAdFPN835WZP25u5U9LB/AJ1p++1dv67Gv99qVzbVW/rTrXvmSL/Bn71ClJkiRJlXPqlCRJkqTKGTQkSZIkVc6gIUmSJKlyBg1JkiRJlTNoSJK0ARFxWERcGRFzIiIj4vgm9Dk1Iv4cEUsj4vGI+FlE7N6Efs+IiL9HxJLydUtEHNPofutqmFr+nL/Q4H7OLfupfT3WyD7LfneIiO9ExMKIWBERd0TEAQ3uc0YX55oR8ZUG99s/Iv4zIh6OiJUR8VBEfDwi/P1zM23s/SgiToyIX0XEgnL9fq2qFQwakiRtzDDgb8A7m9jn4cBXgEMovtiwP3BtRAxrcL+PAh8CDixfvwOuiIi9G9wvABFxEHAa8Pdm9AdMp3j0cudrn0Z2FhHbADcBq4FjgL2A9wGLGtkvcBBPPc/OL8v8YYP7/SDwdor/d/YEPgCcDfxHg/vtzTb2fjSM4t/Yh5pW0Qb4PRqSJG1AZl4DXAMQEc3q82W1yxHxVuBx4ADghgb2e2Vd00ci4gyKwDO9Uf0CRMRw4LvA/wM+2si+aqzJzIaPYtT4IDArM99a0zaj0Z1m5vza5Yj4EPAgcH2Du34+cEVmXlUuz4iIN1CEWG2Gjb0fZea3y3U7NbOu7jiiIUnSlm9U+ecTzeowIvpFxOspPiG9pQldfgW4KjN/04S+Oj2rnILycER8PyJ2aXB/rwRui4gfllPi/hoR/6/BfT5FRAwE3gxclI3/MrU/AC+OiN3KvvcFXghc3eB+tYVwREOSpC1YFB9bng/8ITPvakJ/+1AEi8HAMuCEzLy7wX2+nmK0ppmfdP8JOBm4H9ieYhTl5ojYOzMXNqjPXYAzKP4+PwM8D/hSRHRk5qUN6rPe8cBo4JIm9HUeRUi+NyLWAv2Aj2Tm95rQt7YABg1JkrZs/ws8h+KT4Ga4D9iP4pfRVwPfiojDGxU2ImIS8EXg6Mxsb0QfXSmnoHS6MyJuoZhO9BaKINAIbcBtmfnhcvmv5f0vZwDNChqnAtdk5pwm9PU6itGTN1JMvdsP+EJEzMnMbzWhf7WYQUOSpC1URHyZYrrNYZn5aDP6zMxVwD/KxdvKG7TfDZzeoC4PAMYBt9fMOe8HHBYR7wQGZebaBvX9T5m5PCLuBJ7VwG7mAvWB7R6KQNdwEfEM4CXAic3oD/hv4LOZ+f1y+c6yhqmAQaMPMGhIkrSFKadLfRk4ATgiMx9uZTnAoAYe/7es/7Sni4F7gfOaETIAImIQxZORbmxgNzcB9Y8p3g14pIF91up8qMBVG9uwIkOBdXVta/Ee4T7DoCFJ0gaUT0PataZp5/LZ9E9k5swGdfsViukmrwKWRsT4sn1xZq5sUJ9ExGconmgzCxgBvB44AnjZBnbrkcxcCjzl3pOIWA4sbOQ9KRHxOeBKYCbFiMpHgZE09pP2z1PcB/Jh4AcU92icVr4aqvzuircC38rMNY3ur3QlxZPLZlJMndofeC9wUZP673U29n4UEdsCk4GJ5frdy5HCx5r8hDUAovEPHJAkaesVEUcAv+9i1bcyc0qD+uzu4vzWzLykEX2W/V4IvJjiuxYWU3yfxXmZ+etG9dlNHdcBd2TmWQ3s4/vAYcBYYD7wR+BjTbjx/ThgGsUUrYeB8zPzG43ss+z3aOBXwO6ZeX+j+yv7HAF8imJkbhwwB/ge8Mlyip6epo29H0XEFIoRwXqfyMxzG1halwwakiRJkirnHDlJkiRJlTNoSJIkSaqcQUOSJElS5QwakiRJkipn0JAkSZJUOYOGJEmSpMoZNCRJkiRVzqAhSZIkqXIGDUmSJEmVM2hIkiRJqpxBQ5IkSVLlDBqSJEmSKvf/ATivaVgzJLF9AAAAAElFTkSuQmCC
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we can see more clearly that families that have 2 to 4 members have higher survival chances than bigger families or passengers alone.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[73]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">table</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">crosstab</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Family size&#39;</span><span class="p">],</span> <span class="n">titanic_df</span><span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">])</span>
<span class="nb">print</span><span class="p">(</span><span class="s1">&#39;</span><span class="se">\n</span><span class="s1">&#39;</span><span class="p">,</span> <span class="n">table</span><span class="p">)</span>
<span class="n">table_fractions</span> <span class="o">=</span> <span class="n">table</span><span class="o">.</span><span class="n">div</span><span class="p">(</span><span class="n">table</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="mi">1</span><span class="p">)</span><span class="o">.</span><span class="n">astype</span><span class="p">(</span><span class="nb">float</span><span class="p">),</span> <span class="n">axis</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="n">g</span> <span class="o">=</span> <span class="n">table_fractions</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">kind</span><span class="o">=</span><span class="s2">&quot;bar&quot;</span><span class="p">,</span> <span class="n">stacked</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xticks</span><span class="p">(</span><span class="n">rotation</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">xlabel</span><span class="p">(</span><span class="s1">&#39;Family size&#39;</span><span class="p">,</span> <span class="n">weight</span><span class="o">=</span><span class="s1">&#39;bold&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">ylabel</span><span class="p">(</span><span class="s1">&#39;%&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">tight_layout</span><span class="p">()</span>
<span class="n">leg</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">(</span><span class="n">title</span><span class="o">=</span><span class="s1">&#39;Sex&#39;</span><span class="p">,</span> <span class="n">loc</span><span class="o">=</span><span class="mi">9</span><span class="p">,</span> <span class="n">bbox_to_anchor</span><span class="o">=</span><span class="p">(</span><span class="mf">1.05</span><span class="p">,</span> <span class="mf">1.0</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>
 Sex          Female  Male
Family size              
1               126   411
2                87    74
3                49    53
4                19    10
5                12     3
6                 8    14
7                 8     4
8                 2     4
11                3     4
</pre>
</div>
</div>

<div class="output_area">

    <div class="prompt"></div>




<div class="output_png output_subarea ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Here we can see that the proportion of males is greatest for family size 1 (males traveling alone), and drops for family size of 2, 3, and 4 members (more females). This can explain the survival chances of those groups.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Summary-of-the-EDA">Summary of the EDA<a class="anchor-link" href="#Summary-of-the-EDA">&#182;</a></h2><ul>
<li>Women are more likely to have survived.</li>
<li>Many of the children survived.</li>
<li>Families (2 to 4 members) have better chances of survival.</li>
<li>There are clear relationships between sex, gender, and socio-economic status.</li>
</ul>

</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Models">Models<a class="anchor-link" href="#Models">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<p>Now we are going to be testing different models to make the predictions of the survival. To do this, we prepare the survival values (y) and select a subset of columns to train the models.</p>
<p>The models will be trained using Grid Search Cross Validation to get better performance. In addition, they will be tested using samples that were separated before the training.</p>

</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[74]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">[</span><span class="s2">&quot;Survived&quot;</span><span class="p">])</span><span class="o">.</span><span class="n">ravel</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[75]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">titanic_df</span><span class="o">.</span><span class="n">columns</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[75]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>Index([&#39;Surname&#39;, &#39;Title&#39;, &#39;Name&#39;, &#39;Sex&#39;, &#39;Age&#39;, &#39;Completed age&#39;,
       &#39;Discrete age&#39;, &#39;Categorized age&#39;, &#39;Economic status&#39;, &#39;Fare&#39;,
       &#39;Individual fare&#39;, &#39;Number of siblings/spouses&#39;,
       &#39;Number of parents/children&#39;, &#39;Family size&#39;, &#39;Cabin&#39;, &#39;Embarked&#39;,
       &#39;Ticket&#39;, &#39;Survived&#39;],
      dtype=&#39;object&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[76]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">subset_categorical_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Sex&#39;</span><span class="p">,</span><span class="s1">&#39;Categorized age&#39;</span><span class="p">,</span><span class="s1">&#39;Economic status&#39;</span><span class="p">,</span><span class="s1">&#39;Embarked&#39;</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[77]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">subset_numerical_columns</span> <span class="o">=</span> <span class="p">[</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span><span class="s1">&#39;Fare&#39;</span><span class="p">,</span><span class="s1">&#39;Family size&#39;</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[78]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span> <span class="o">=</span> <span class="p">{}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="KNN">KNN<a class="anchor-link" href="#KNN">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[79]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;n_neighbors&#39;</span><span class="p">:[</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">4</span><span class="p">,</span><span class="mi">5</span><span class="p">,</span><span class="mi">6</span><span class="p">,</span><span class="mi">7</span><span class="p">,</span><span class="mi">8</span><span class="p">,</span><span class="mi">9</span><span class="p">,</span><span class="mi">10</span><span class="p">,</span><span class="mi">15</span><span class="p">,</span><span class="mi">20</span><span class="p">,</span><span class="mi">25</span><span class="p">,</span><span class="mi">30</span><span class="p">],</span>
    <span class="s1">&#39;weights&#39;</span><span class="p">:[</span><span class="s1">&#39;uniform&#39;</span><span class="p">,</span> <span class="s1">&#39;distance&#39;</span><span class="p">],</span>
    <span class="s1">&#39;metric&#39;</span><span class="p">:[</span><span class="s1">&#39;minkowski&#39;</span><span class="p">,</span><span class="s1">&#39;cosine&#39;</span><span class="p">,</span><span class="s1">&#39;chebyshev&#39;</span><span class="p">,</span><span class="s1">&#39;correlation&#39;</span><span class="p">]</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[80]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeDataset</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[81]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[82]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">knn</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">KNeighborsClassifier</span><span class="p">(),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[83]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">knn</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[83]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=KNeighborsClassifier(), n_jobs=-1,
             param_grid={&#39;metric&#39;: [&#39;minkowski&#39;, &#39;cosine&#39;, &#39;chebyshev&#39;,
                                    &#39;correlation&#39;],
                         &#39;n_neighbors&#39;: [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25,
                                         30],
                         &#39;weights&#39;: [&#39;uniform&#39;, &#39;distance&#39;]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[84]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">knn</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[85]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">knn</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[85]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;metric&#39;: &#39;cosine&#39;, &#39;n_neighbors&#39;: 3, &#39;weights&#39;: &#39;uniform&#39;}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[86]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.76      0.85      0.80       110
           1       0.70      0.58      0.63        69

    accuracy                           0.74       179
   macro avg       0.73      0.71      0.72       179
weighted avg       0.74      0.74      0.74       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[87]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;KNN&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="SVM">SVM<a class="anchor-link" href="#SVM">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[88]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;C&#39;</span><span class="p">:[</span><span class="mf">0.01</span><span class="p">,</span><span class="mf">0.1</span><span class="p">,</span><span class="mf">0.5</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">5</span><span class="p">,</span><span class="mi">10</span><span class="p">,</span><span class="mi">15</span><span class="p">,</span><span class="mi">20</span><span class="p">,</span><span class="mi">25</span><span class="p">,</span><span class="mi">30</span><span class="p">],</span>
    <span class="s1">&#39;kernel&#39;</span><span class="p">:[</span><span class="s2">&quot;poly&quot;</span><span class="p">,</span> <span class="s2">&quot;rbf&quot;</span><span class="p">,</span> <span class="s2">&quot;linear&quot;</span><span class="p">],</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[89]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeAndNormalizeData</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[90]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[91]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">svm</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">SVC</span><span class="p">(),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[92]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">svm</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[92]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=SVC(), n_jobs=-1,
             param_grid={&#39;C&#39;: [0.01, 0.1, 0.5, 1, 5, 10, 15, 20, 25, 30],
                         &#39;kernel&#39;: [&#39;poly&#39;, &#39;rbf&#39;, &#39;linear&#39;]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[93]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">svm</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[94]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">svm</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[94]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;C&#39;: 0.5, &#39;kernel&#39;: &#39;poly&#39;}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[95]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.84      0.88      0.86       110
           1       0.79      0.72      0.76        69

    accuracy                           0.82       179
   macro avg       0.81      0.80      0.81       179
weighted avg       0.82      0.82      0.82       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[96]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;SVM&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Decision-Tree-Classifier">Decision Tree Classifier<a class="anchor-link" href="#Decision-Tree-Classifier">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[97]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;criterion&#39;</span><span class="p">:[</span><span class="s1">&#39;gini&#39;</span><span class="p">,</span> <span class="s1">&#39;entropy&#39;</span><span class="p">],</span>
    <span class="s1">&#39;max_depth&#39;</span><span class="p">:[</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">4</span><span class="p">,</span><span class="mi">5</span><span class="p">,</span><span class="mi">6</span><span class="p">,</span><span class="mi">7</span><span class="p">,</span><span class="mi">8</span><span class="p">,</span><span class="mi">9</span><span class="p">,</span><span class="mi">10</span><span class="p">],</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[98]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeDataset</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[99]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[100]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dtc</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">DecisionTreeClassifier</span><span class="p">(),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[101]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dtc</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[101]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=DecisionTreeClassifier(), n_jobs=-1,
             param_grid={&#39;criterion&#39;: [&#39;gini&#39;, &#39;entropy&#39;],
                         &#39;max_depth&#39;: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[102]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">dtc</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[103]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">dtc</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[103]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;criterion&#39;: &#39;entropy&#39;, &#39;max_depth&#39;: 5}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[104]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.85      0.85      0.85       110
           1       0.76      0.75      0.76        69

    accuracy                           0.82       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.82      0.82      0.82       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[105]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Decision Tree Classifier&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Logistic-Regression">Logistic Regression<a class="anchor-link" href="#Logistic-Regression">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[106]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeAndNormalizeData</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[107]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;penalty&#39;</span><span class="p">:[</span><span class="s2">&quot;l2&quot;</span><span class="p">,</span><span class="s2">&quot;none&quot;</span><span class="p">,</span> <span class="s2">&quot;l1&quot;</span><span class="p">,</span> <span class="s2">&quot;elasticnet&quot;</span><span class="p">],</span>
    <span class="s1">&#39;C&#39;</span><span class="p">:[</span><span class="mf">0.001</span><span class="p">,</span><span class="mf">0.01</span><span class="p">,</span><span class="mf">0.1</span><span class="p">,</span><span class="mf">0.2</span><span class="p">,</span><span class="mf">0.5</span><span class="p">,</span><span class="mf">0.7</span><span class="p">,</span><span class="mf">1.0</span><span class="p">,</span><span class="mf">1.5</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mf">2.5</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mf">3.5</span><span class="p">,</span><span class="mi">4</span><span class="p">,</span><span class="mi">5</span><span class="p">,</span><span class="mi">6</span><span class="p">,</span><span class="mi">7</span><span class="p">,</span><span class="mi">10</span><span class="p">],</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[108]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[109]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">lr</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">LogisticRegression</span><span class="p">(</span><span class="n">solver</span><span class="o">=</span><span class="s1">&#39;saga&#39;</span><span class="p">,</span><span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">,</span><span class="n">max_iter</span><span class="o">=</span><span class="mi">700</span><span class="p">),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[110]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">lr</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[110]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=LogisticRegression(max_iter=700, random_state=0,
                                          solver=&#39;saga&#39;),
             n_jobs=-1,
             param_grid={&#39;C&#39;: [0.001, 0.01, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2,
                               2.5, 3, 3.5, 4, 5, 6, 7, 10],
                         &#39;penalty&#39;: [&#39;l2&#39;, &#39;none&#39;, &#39;l1&#39;, &#39;elasticnet&#39;]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[111]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">lr</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[112]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">lr</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[112]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;C&#39;: 5, &#39;penalty&#39;: &#39;l2&#39;}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[113]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.84      0.86      0.85       110
           1       0.77      0.74      0.76        69

    accuracy                           0.82       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.82      0.81       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[114]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Logistic Regression&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Naive-Bayes">Naive Bayes<a class="anchor-link" href="#Naive-Bayes">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Categorical-NB">Categorical NB<a class="anchor-link" href="#Categorical-NB">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[115]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;alpha&#39;</span><span class="p">:[</span><span class="mf">0.001</span><span class="p">,</span><span class="mf">0.01</span><span class="p">,</span><span class="mf">0.1</span><span class="p">,</span><span class="mf">0.3</span><span class="p">,</span><span class="mf">0.5</span><span class="p">,</span><span class="mf">0.7</span><span class="p">,</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">10</span><span class="p">],</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[116]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeDataset</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,[])</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[117]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[118]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cnb</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">CategoricalNB</span><span class="p">(),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[119]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cnb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[119]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=CategoricalNB(), n_jobs=-1,
             param_grid={&#39;alpha&#39;: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 2, 3,
                                   10]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[120]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">cnb</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[121]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">cnb</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[121]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;alpha&#39;: 0.001}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[122]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.82      0.84      0.83       110
           1       0.73      0.71      0.72        69

    accuracy                           0.79       179
   macro avg       0.78      0.77      0.77       179
weighted avg       0.79      0.79      0.79       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[123]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">prob_categorical</span> <span class="o">=</span> <span class="n">cnb</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span>
<span class="n">prob_x_test_categorical</span> <span class="o">=</span> <span class="n">cnb</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[124]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Categorical NB&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Multinomial-NB">Multinomial NB<a class="anchor-link" href="#Multinomial-NB">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[125]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeDataset</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,[</span><span class="s1">&#39;Sex&#39;</span><span class="p">,</span><span class="s1">&#39;Discrete age&#39;</span><span class="p">,</span><span class="s1">&#39;Completed age&#39;</span><span class="p">,</span><span class="s1">&#39;Economic status&#39;</span><span class="p">],[</span><span class="s1">&#39;Family size&#39;</span><span class="p">])</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[126]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[127]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">mnb</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">MultinomialNB</span><span class="p">(),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[128]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">mnb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[128]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=MultinomialNB(), n_jobs=-1,
             param_grid={&#39;alpha&#39;: [0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1, 2, 3,
                                   10]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[129]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">mnb</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[130]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">mnb</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[130]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;alpha&#39;: 10}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[131]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_pred</span><span class="p">,</span><span class="n">y_test</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.93      0.74      0.83       137
           1       0.49      0.81      0.61        42

    accuracy                           0.76       179
   macro avg       0.71      0.78      0.72       179
weighted avg       0.83      0.76      0.78       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[132]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">prob_multinomial</span> <span class="o">=</span> <span class="n">mnb</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span>
<span class="n">prob_x_test_multinomial</span> <span class="o">=</span> <span class="n">mnb</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[133]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Multinomial NB&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Gaussian-NB">Gaussian NB<a class="anchor-link" href="#Gaussian-NB">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[134]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;var_smoothing&#39;</span><span class="p">:[</span><span class="mf">0.00000001</span><span class="p">,</span><span class="mf">0.0000001</span><span class="p">,</span><span class="mf">0.000001</span><span class="p">,</span><span class="mf">0.00001</span><span class="p">,</span><span class="mf">0.0001</span><span class="p">,</span><span class="mf">0.001</span><span class="p">],</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[135]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeDataset</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,[],[</span><span class="s1">&#39;Age&#39;</span><span class="p">,</span><span class="s1">&#39;Fare&#39;</span><span class="p">,</span><span class="s1">&#39;Individual fare&#39;</span><span class="p">])</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[136]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[137]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">gnb</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">GaussianNB</span><span class="p">(),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[138]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">gnb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[138]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=GaussianNB(), n_jobs=-1,
             param_grid={&#39;var_smoothing&#39;: [1e-08, 1e-07, 1e-06, 1e-05, 0.0001,
                                           0.001]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[139]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">gnb</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[140]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">gnb</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[140]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;var_smoothing&#39;: 1e-08}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[141]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.69      0.96      0.80       110
           1       0.84      0.30      0.45        69

    accuracy                           0.71       179
   macro avg       0.76      0.63      0.62       179
weighted avg       0.75      0.71      0.67       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[142]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">prob_gaussian</span> <span class="o">=</span> <span class="n">gnb</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span>
<span class="n">prob_x_test_gaussian</span> <span class="o">=</span> <span class="n">gnb</span><span class="o">.</span><span class="n">predict_proba</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[143]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Gaussian NB&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h4 id="Ensemble-NB">Ensemble NB<a class="anchor-link" href="#Ensemble-NB">&#182;</a></h4>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[144]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">prob_x_train</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">hstack</span><span class="p">((</span><span class="n">prob_multinomial</span><span class="p">,</span> <span class="n">prob_categorical</span> <span class="p">,</span> <span class="n">prob_gaussian</span><span class="p">))</span>
<span class="n">prob_x_test</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">hstack</span><span class="p">((</span><span class="n">prob_x_test_multinomial</span><span class="p">,</span> <span class="n">prob_x_test_categorical</span> <span class="p">,</span> <span class="n">prob_x_test_gaussian</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[145]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">ensemble_nb</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">GaussianNB</span><span class="p">(),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[146]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">ensemble_nb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">prob_x_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[146]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=GaussianNB(), n_jobs=-1,
             param_grid={&#39;var_smoothing&#39;: [1e-08, 1e-07, 1e-06, 1e-05, 0.0001,
                                           0.001]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[147]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">ensemble_nb</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">prob_x_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[148]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">ensemble_nb</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[148]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;var_smoothing&#39;: 1e-08}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[149]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.83      0.78      0.81       110
           1       0.68      0.75      0.72        69

    accuracy                           0.77       179
   macro avg       0.76      0.77      0.76       179
weighted avg       0.78      0.77      0.77       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[150]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;NB Ensemble&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Boosting">Boosting<a class="anchor-link" href="#Boosting">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[151]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeDataset</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[152]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;criterion&#39;</span><span class="p">:[</span><span class="s2">&quot;friedman_mse&quot;</span><span class="p">,</span> <span class="s2">&quot;mse&quot;</span><span class="p">,</span> <span class="s2">&quot;mae&quot;</span><span class="p">],</span>
    <span class="s1">&#39;loss&#39;</span><span class="p">:[</span><span class="s2">&quot;deviance&quot;</span><span class="p">,</span> <span class="s2">&quot;exponential&quot;</span><span class="p">],</span>
    <span class="s1">&#39;learning_rate&#39;</span><span class="p">:[</span><span class="mf">0.1</span><span class="p">,</span> <span class="mf">0.01</span><span class="p">,</span> <span class="mf">0.001</span><span class="p">],</span>
    <span class="s1">&#39;n_estimators&#39;</span><span class="p">:[</span><span class="mi">5</span><span class="p">,</span><span class="mi">9</span><span class="p">,</span><span class="mi">10</span><span class="p">,</span><span class="mi">11</span><span class="p">,</span><span class="mi">12</span><span class="p">,</span><span class="mi">13</span><span class="p">,</span><span class="mi">14</span><span class="p">,</span><span class="mi">15</span><span class="p">,</span><span class="mi">20</span><span class="p">,</span><span class="mi">30</span><span class="p">,</span><span class="mi">40</span><span class="p">,</span><span class="mi">50</span><span class="p">,</span><span class="mi">60</span><span class="p">,</span><span class="mi">100</span><span class="p">]</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[153]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[154]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">gb</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">GradientBoostingClassifier</span><span class="p">(</span><span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[155]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">gb</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[155]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=GradientBoostingClassifier(random_state=0), n_jobs=-1,
             param_grid={&#39;criterion&#39;: [&#39;friedman_mse&#39;, &#39;mse&#39;, &#39;mae&#39;],
                         &#39;learning_rate&#39;: [0.1, 0.01, 0.001],
                         &#39;loss&#39;: [&#39;deviance&#39;, &#39;exponential&#39;],
                         &#39;n_estimators&#39;: [5, 9, 10, 11, 12, 13, 14, 15, 20, 30,
                                          40, 50, 60, 100]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[156]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">gb</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[157]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">gb</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[157]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;criterion&#39;: &#39;friedman_mse&#39;,
 &#39;learning_rate&#39;: 0.1,
 &#39;loss&#39;: &#39;deviance&#39;,
 &#39;n_estimators&#39;: 15}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[158]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.85      0.88      0.87       110
           1       0.80      0.75      0.78        69

    accuracy                           0.83       179
   macro avg       0.83      0.82      0.82       179
weighted avg       0.83      0.83      0.83       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[159]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Gradient Boosting&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Random-Forest">Random Forest<a class="anchor-link" href="#Random-Forest">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[160]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeAndNormalizeData</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[161]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">params</span> <span class="o">=</span> <span class="p">{</span>
    <span class="s1">&#39;criterion&#39;</span><span class="p">:[</span><span class="s2">&quot;gini&quot;</span><span class="p">,</span> <span class="s2">&quot;entropy&quot;</span><span class="p">],</span>
    <span class="s1">&#39;max_depth&#39;</span><span class="p">:[</span><span class="mi">1</span><span class="p">,</span><span class="mi">2</span><span class="p">,</span><span class="mi">3</span><span class="p">,</span><span class="mi">4</span><span class="p">,</span><span class="mi">5</span><span class="p">,</span><span class="mi">6</span><span class="p">,</span><span class="mi">7</span><span class="p">,</span><span class="mi">9</span><span class="p">,</span><span class="mi">10</span><span class="p">],</span>
    <span class="s1">&#39;n_estimators&#39;</span><span class="p">:[</span><span class="mi">5</span><span class="p">,</span><span class="mi">9</span><span class="p">,</span><span class="mi">10</span><span class="p">,</span><span class="mi">11</span><span class="p">,</span><span class="mi">12</span><span class="p">,</span><span class="mi">13</span><span class="p">,</span><span class="mi">14</span><span class="p">,</span><span class="mi">15</span><span class="p">,</span><span class="mi">20</span><span class="p">,</span><span class="mi">30</span><span class="p">,</span><span class="mi">40</span><span class="p">,</span><span class="mi">50</span><span class="p">,</span><span class="mi">60</span><span class="p">,</span><span class="mi">100</span><span class="p">]</span>
<span class="p">}</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[162]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[163]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">rf</span> <span class="o">=</span> <span class="n">GridSearchCV</span><span class="p">(</span><span class="n">RandomForestClassifier</span><span class="p">(</span><span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">),</span> <span class="n">params</span><span class="p">,</span> <span class="n">scoring</span> <span class="o">=</span> <span class="s1">&#39;accuracy&#39;</span><span class="p">,</span> <span class="n">n_jobs</span> <span class="o">=</span> <span class="o">-</span><span class="mi">1</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[164]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">rf</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[164]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>GridSearchCV(estimator=RandomForestClassifier(random_state=0), n_jobs=-1,
             param_grid={&#39;criterion&#39;: [&#39;gini&#39;, &#39;entropy&#39;],
                         &#39;max_depth&#39;: [1, 2, 3, 4, 5, 6, 7, 9, 10],
                         &#39;n_estimators&#39;: [5, 9, 10, 11, 12, 13, 14, 15, 20, 30,
                                          40, 50, 60, 100]},
             scoring=&#39;accuracy&#39;)</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[165]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">rf</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[166]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">rf</span><span class="o">.</span><span class="n">best_params_</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[166]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>{&#39;criterion&#39;: &#39;gini&#39;, &#39;max_depth&#39;: 6, &#39;n_estimators&#39;: 9}</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[167]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.82      0.91      0.86       110
           1       0.82      0.68      0.75        69

    accuracy                           0.82       179
   macro avg       0.82      0.80      0.80       179
weighted avg       0.82      0.82      0.82       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[168]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Random Forest&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Voting">Voting<a class="anchor-link" href="#Voting">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[169]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeAndNormalizeData</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[170]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[171]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">v</span> <span class="o">=</span> <span class="n">VotingClassifier</span><span class="p">(</span><span class="n">estimators</span><span class="o">=</span><span class="p">[(</span><span class="s1">&#39;lr&#39;</span><span class="p">,</span> <span class="n">lr</span><span class="p">),</span> <span class="p">(</span><span class="s1">&#39;rf&#39;</span><span class="p">,</span> <span class="n">rf</span><span class="p">),(</span><span class="s1">&#39;svm&#39;</span><span class="p">,</span><span class="n">svm</span><span class="p">),(</span><span class="s1">&#39;tree&#39;</span><span class="p">,</span><span class="n">dtc</span><span class="p">)],</span> <span class="n">voting</span><span class="o">=</span><span class="s1">&#39;hard&#39;</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[172]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">v</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[172]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>VotingClassifier(estimators=[(&#39;lr&#39;,
                              GridSearchCV(estimator=LogisticRegression(max_iter=700,
                                                                        random_state=0,
                                                                        solver=&#39;saga&#39;),
                                           n_jobs=-1,
                                           param_grid={&#39;C&#39;: [0.001, 0.01, 0.1,
                                                             0.2, 0.5, 0.7, 1.0,
                                                             1.5, 2, 2.5, 3,
                                                             3.5, 4, 5, 6, 7,
                                                             10],
                                                       &#39;penalty&#39;: [&#39;l2&#39;, &#39;none&#39;,
                                                                   &#39;l1&#39;,
                                                                   &#39;elasticnet&#39;]},
                                           scoring=&#39;accuracy&#39;)),
                             (&#39;rf&#39;,
                              GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                                           n_jobs=-1,
                                           pa...
                                                                        40, 50,
                                                                        60,
                                                                        100]},
                                           scoring=&#39;accuracy&#39;)),
                             (&#39;svm&#39;,
                              GridSearchCV(estimator=SVC(), n_jobs=-1,
                                           param_grid={&#39;C&#39;: [0.01, 0.1, 0.5, 1,
                                                             5, 10, 15, 20, 25,
                                                             30],
                                                       &#39;kernel&#39;: [&#39;poly&#39;, &#39;rbf&#39;,
                                                                  &#39;linear&#39;]},
                                           scoring=&#39;accuracy&#39;)),
                             (&#39;tree&#39;,
                              GridSearchCV(estimator=DecisionTreeClassifier(),
                                           n_jobs=-1,
                                           param_grid={&#39;criterion&#39;: [&#39;gini&#39;,
                                                                     &#39;entropy&#39;],
                                                       &#39;max_depth&#39;: [1, 2, 3, 4,
                                                                     5, 6, 7, 8,
                                                                     9, 10]},
                                           scoring=&#39;accuracy&#39;))])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[173]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">v</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[174]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.82      0.90      0.86       110
           1       0.81      0.70      0.75        69

    accuracy                           0.82       179
   macro avg       0.82      0.80      0.81       179
weighted avg       0.82      0.82      0.82       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[175]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Voting&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Stacking">Stacking<a class="anchor-link" href="#Stacking">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[176]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X</span> <span class="o">=</span> <span class="n">encodeAndNormalizeData</span><span class="p">(</span><span class="n">titanic_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[177]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_train</span><span class="p">,</span> <span class="n">X_test</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">train_test_split</span><span class="p">(</span><span class="n">X</span><span class="p">,</span> <span class="n">y</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[178]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">s</span> <span class="o">=</span> <span class="n">StackingClassifier</span><span class="p">(</span><span class="n">estimators</span><span class="o">=</span><span class="p">[(</span><span class="s1">&#39;lr&#39;</span><span class="p">,</span> <span class="n">lr</span><span class="p">),</span> <span class="p">(</span><span class="s1">&#39;rf&#39;</span><span class="p">,</span> <span class="n">rf</span><span class="p">),(</span><span class="s1">&#39;svm&#39;</span><span class="p">,</span><span class="n">svm</span><span class="p">),(</span><span class="s1">&#39;tree&#39;</span><span class="p">,</span><span class="n">dtc</span><span class="p">)])</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[179]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">s</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[179]:</div>




<div class="output_text output_subarea output_execute_result">
<pre>StackingClassifier(estimators=[(&#39;lr&#39;,
                                GridSearchCV(estimator=LogisticRegression(max_iter=700,
                                                                          random_state=0,
                                                                          solver=&#39;saga&#39;),
                                             n_jobs=-1,
                                             param_grid={&#39;C&#39;: [0.001, 0.01, 0.1,
                                                               0.2, 0.5, 0.7,
                                                               1.0, 1.5, 2, 2.5,
                                                               3, 3.5, 4, 5, 6,
                                                               7, 10],
                                                         &#39;penalty&#39;: [&#39;l2&#39;,
                                                                     &#39;none&#39;,
                                                                     &#39;l1&#39;,
                                                                     &#39;elasticnet&#39;]},
                                             scoring=&#39;accuracy&#39;)),
                               (&#39;rf&#39;,
                                GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                                             n_jobs=-1,...
                                                                          50,
                                                                          60,
                                                                          100]},
                                             scoring=&#39;accuracy&#39;)),
                               (&#39;svm&#39;,
                                GridSearchCV(estimator=SVC(), n_jobs=-1,
                                             param_grid={&#39;C&#39;: [0.01, 0.1, 0.5,
                                                               1, 5, 10, 15, 20,
                                                               25, 30],
                                                         &#39;kernel&#39;: [&#39;poly&#39;,
                                                                    &#39;rbf&#39;,
                                                                    &#39;linear&#39;]},
                                             scoring=&#39;accuracy&#39;)),
                               (&#39;tree&#39;,
                                GridSearchCV(estimator=DecisionTreeClassifier(),
                                             n_jobs=-1,
                                             param_grid={&#39;criterion&#39;: [&#39;gini&#39;,
                                                                       &#39;entropy&#39;],
                                                         &#39;max_depth&#39;: [1, 2, 3,
                                                                       4, 5, 6,
                                                                       7, 8, 9,
                                                                       10]},
                                             scoring=&#39;accuracy&#39;))])</pre>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[180]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">s</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[181]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">))</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>              precision    recall  f1-score   support

           0       0.83      0.90      0.86       110
           1       0.82      0.71      0.76        69

    accuracy                           0.83       179
   macro avg       0.82      0.81      0.81       179
weighted avg       0.83      0.83      0.82       179

</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[182]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">metrics</span><span class="p">[</span><span class="s2">&quot;Stacking&quot;</span><span class="p">]</span> <span class="o">=</span> <span class="n">classification_report</span><span class="p">(</span><span class="n">y_test</span><span class="p">,</span><span class="n">y_pred</span><span class="p">,</span><span class="n">output_dict</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h3 id="Results">Results<a class="anchor-link" href="#Results">&#182;</a></h3>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[183]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">results</span> <span class="o">=</span> <span class="p">[(</span><span class="n">model</span><span class="p">,</span><span class="n">metrics</span><span class="p">[</span><span class="n">model</span><span class="p">][</span><span class="s1">&#39;accuracy&#39;</span><span class="p">])</span> <span class="k">for</span> <span class="n">model</span> <span class="ow">in</span> <span class="n">metrics</span><span class="p">]</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[184]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">results_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">results</span><span class="p">,</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="s2">&quot;Model&quot;</span><span class="p">,</span><span class="s2">&quot;Accuracy&quot;</span><span class="p">])</span>
<span class="n">results_df</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="n">by</span><span class="o">=</span><span class="s2">&quot;Accuracy&quot;</span><span class="p">,</span><span class="n">ascending</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[184]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>Gradient Boosting</td>
      <td>0.832402</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Stacking</td>
      <td>0.826816</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVM</td>
      <td>0.821229</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Random Forest</td>
      <td>0.821229</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Voting</td>
      <td>0.821229</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Decision Tree Classifier</td>
      <td>0.815642</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Logistic Regression</td>
      <td>0.815642</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Categorical NB</td>
      <td>0.787709</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NB Ensemble</td>
      <td>0.770950</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Multinomial NB</td>
      <td>0.759777</td>
    </tr>
    <tr>
      <th>0</th>
      <td>KNN</td>
      <td>0.743017</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Gaussian NB</td>
      <td>0.709497</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="Predictions">Predictions<a class="anchor-link" href="#Predictions">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[185]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">test_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s2">&quot;Datasets/test.csv&quot;</span><span class="p">)</span>
<span class="n">test_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[185]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[186]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">test_df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt"></div>


<div class="output_subarea output_stream output_stdout output_text">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  418 non-null    int64  
 1   Pclass       418 non-null    int64  
 2   Name         418 non-null    object 
 3   Sex          418 non-null    object 
 4   Age          332 non-null    float64
 5   SibSp        418 non-null    int64  
 6   Parch        418 non-null    int64  
 7   Ticket       418 non-null    object 
 8   Fare         417 non-null    float64
 9   Cabin        91 non-null     object 
 10  Embarked     418 non-null    object 
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
</pre>
</div>
</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[187]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">prepareTestDataset</span><span class="p">(</span><span class="n">test_df</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[188]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">X_test</span> <span class="o">=</span> <span class="n">encodeDataset</span><span class="p">(</span><span class="n">test_df</span><span class="p">,</span><span class="n">subset_categorical_columns</span><span class="p">,</span><span class="n">subset_numerical_columns</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[189]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">test_df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

<div class="output_wrapper">
<div class="output">


<div class="output_area">

    <div class="prompt output_prompt">Out[189]:</div>



<div class="output_html rendered_html output_subarea output_execute_result">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Economic status</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>Number of siblings/spouses</th>
      <th>Number of parents/children</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Surname</th>
      <th>Title</th>
      <th>Completed age</th>
      <th>Ticket size</th>
      <th>Individual fare</th>
      <th>Family size</th>
      <th>Discrete age</th>
      <th>Categorized age</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>Lower</td>
      <td>Kelly, Mr. James</td>
      <td>Male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>-</td>
      <td>Queenstown</td>
      <td>Kelly</td>
      <td>Mr</td>
      <td>False</td>
      <td>1</td>
      <td>7.8292</td>
      <td>1</td>
      <td>(30, 35]</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>893</th>
      <td>Lower</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>Female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>Wilkes</td>
      <td>Mrs</td>
      <td>False</td>
      <td>1</td>
      <td>7.0000</td>
      <td>2</td>
      <td>(45, 50]</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>894</th>
      <td>Middle</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>Male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>-</td>
      <td>Queenstown</td>
      <td>Myles</td>
      <td>Mr</td>
      <td>False</td>
      <td>1</td>
      <td>9.6875</td>
      <td>1</td>
      <td>(60, 65]</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>895</th>
      <td>Lower</td>
      <td>Wirz, Mr. Albert</td>
      <td>Male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>Wirz</td>
      <td>Mr</td>
      <td>False</td>
      <td>1</td>
      <td>8.6625</td>
      <td>1</td>
      <td>(25, 30]</td>
      <td>Adult</td>
    </tr>
    <tr>
      <th>896</th>
      <td>Lower</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>Female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>-</td>
      <td>Southhampton</td>
      <td>Hirvonen</td>
      <td>Mrs</td>
      <td>False</td>
      <td>1</td>
      <td>12.2875</td>
      <td>3</td>
      <td>(20, 25]</td>
      <td>Youth</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[190]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">y_pred</span> <span class="o">=</span> <span class="n">gb</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_test</span><span class="p">)</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing code_cell rendered">
<div class="input">
<div class="prompt input_prompt">In&nbsp;[191]:</div>
<div class="inner_cell">
    <div class="input_area">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">submission_file</span> <span class="o">=</span> <span class="nb">open</span><span class="p">(</span><span class="s2">&quot;Datasets/Submission.csv&quot;</span><span class="p">,</span> <span class="s2">&quot;w&quot;</span><span class="p">)</span>
<span class="n">submission_file</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="s2">&quot;PassengerId,Survived</span><span class="se">\n</span><span class="s2">&quot;</span><span class="p">)</span>
<span class="n">i</span> <span class="o">=</span> <span class="mi">0</span>
<span class="k">for</span> <span class="n">prediction</span> <span class="ow">in</span> <span class="n">y_pred</span><span class="p">:</span>
    <span class="n">submission_file</span><span class="o">.</span><span class="n">write</span><span class="p">(</span><span class="nb">str</span><span class="p">(</span><span class="n">test_df</span><span class="o">.</span><span class="n">index</span><span class="p">[</span><span class="n">i</span><span class="p">])</span><span class="o">+</span> <span class="s2">&quot;,&quot;</span> <span class="o">+</span> <span class="nb">str</span><span class="p">(</span><span class="n">prediction</span><span class="p">)</span> <span class="o">+</span> <span class="s2">&quot;</span><span class="se">\n</span><span class="s2">&quot;</span><span class="p">)</span>
    <span class="n">i</span> <span class="o">=</span> <span class="n">i</span> <span class="o">+</span> <span class="mi">1</span>
<span class="n">submission_file</span><span class="o">.</span><span class="n">close</span><span class="p">()</span>
</pre></div>

    </div>
</div>
</div>

</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<h2 id="References">References<a class="anchor-link" href="#References">&#182;</a></h2>
</div>
</div>
</div>
<div class="cell border-box-sizing text_cell rendered"><div class="prompt input_prompt">
</div><div class="inner_cell">
<div class="text_cell_render border-box-sizing rendered_html">
<ul>
<li><a href="https://www.kaggle.com/c/titanic">Kaggle</a></li>
<li><a href="https://rpmarchildon.com/ai-titanic/">A Tour of Machine Learning in Python</a></li>
<li><a href="https://www.kaggle.com/chapagain/titanic-solution-a-beginner-s-guide">Titanic begginer's Guide</a></li>
<li><a href="https://www.encyclopedia-titanica.org/">Encyclopedia-titanica</a>.</li>
</ul>

</div>
</div>
</div>
    </div>
  </div>
</body>

 


</html>
