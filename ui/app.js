"use strict";
const flights = EXPLORER_DATA.flights.map(([number, from, to, origin, destination, date, time, six, one, delay], id) =>
  ({id, number, from, to, origin, destination, date, time, six, one, delay}));
const days = [...new Set(flights.map(f => f.date))].sort();
const routes = new Map();
for (const f of flights) {
  const key = f.from + " → " + f.to;
  if (!routes.has(key)) routes.set(key, []);
  routes.get(key).push(f);
}
const view = document.querySelector("#view");
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = value => Math.round(value * 100) + "%";
const dateLabel = date => new Date(date+"T12:00:00Z").toLocaleDateString("en-US", {month:"short",day:"numeric",year:"numeric",timeZone:"UTC"});
let current = flights.find(f => f.from==="ATL" && f.to==="ORD") || flights[0];
let routeFilter = "", routePage = 0, metric = "pr_auc_mean";
function layout(side, content) {
  view.innerHTML = '<div class="layout"><aside>'+side+'</aside><section>'+content+'</section></div>';
}
function panel(title, label, body) {
  return '<article class="ticket"><div class="strip"><span>'+title+'</span><small>'+label+'</small></div>'+body+'</article>';
}
function showTicket(f) {
  const stop = (code, city) => '<div><strong class="code">'+escapeHTML(code)+'</strong><span class="city">'+escapeHTML(city)+'</span></div>';
  const risk = (hours, value) => '<div class="checkpoint"><b>'+hours+' before<br>departure</b><i class="dot"></i><strong>'+pct(value)+'</strong></div>';
  const body = '<div class="route">'+stop(f.from,f.origin)+'<div class="arc" aria-hidden="true"><span>✈︎</span></div>'+stop(f.to,f.destination)+'</div><div class="perforation"></div><div class="risks"><h2>Chance of departing at least 15 minutes late</h2><div class="timeline">'+risk("6 hours",f.six)+risk("1 hour",f.one)+'</div></div>';
  $("boarding").innerHTML = panel('✈︎ &nbsp; FLIGHT '+escapeHTML(f.number),'Pilot validation',body)+
    '<div class="details"><span>'+dateLabel(f.date)+' · '+f.time+' UTC · LightGBM</span><button class="linkbtn" id="reveal" aria-expanded="false">Reveal outcome</button></div><p class="notice" id="outcome" aria-live="polite"></p>';
  $("reveal").onclick = () => {
    const visible = $("reveal").getAttribute("aria-expanded")==="true";
    const timing = f.delay===0 ? "on time" : Math.abs(f.delay)+" minutes "+(f.delay>0?"late":"early");
    $("outcome").textContent = visible ? "" : "Actual departure: "+timing+". "+(f.delay>=15?"Classified as delayed.":"Below the 15-minute delay threshold.");
    $("reveal").setAttribute("aria-expanded", String(!visible));
    $("reveal").textContent = visible ? "Reveal outcome" : "Hide outcome";
  };
}
function prediction() {
  layout('<h1>Explore a Historical Flight</h1><form id="picker"><label for="flight">Flight</label><select id="flight"></select><label for="date">Date <small>(UTC)</small></label><select id="date"></select><button class="primary">Predict Delay</button></form>', '<div id="boarding" aria-live="polite"></div>');
  $("date").replaceChildren(...days.map(d=>new Option(dateLabel(d),d)));
  $("date").value=current.date;
  function options() {
    $("flight").replaceChildren(...flights.filter(f=>f.date===$("date").value).map(f=>new Option(f.number+" ("+f.from+" → "+f.to+") · "+f.time,f.id)));
  }
  options();
  $("flight").value=current.id;
  $("date").onchange=options;
  $("picker").onsubmit=event=>{event.preventDefault();current=flights[Number($("flight").value)];showTicket(current);};
  showTicket(current);
}
function routeList() {
  layout('<h1>Explore a Route</h1><label for="search">Airport or city</label><input id="search" type="search" placeholder="e.g. ATL or Chicago"><p class="note">Choose a route to open a historical flight on its boarding pass.</p>',
    panel("Flight routes","Pilot validation",'<div class="body" id="routeRows"></div>'));
  $("search").value=routeFilter;
  function display() {
    const query=routeFilter.toLowerCase().trim();
    const items=[...routes].filter(([k,[f]])=>(k+" "+f.origin+" "+f.destination).toLowerCase().includes(query))
      .sort((a,b)=>b[1].length-a[1].length || a[0].localeCompare(b[0]));
    const slice=items.slice(routePage*8,routePage*8+8);
    $("routeRows").innerHTML='<h2>Route directory</h2><p class="caption">'+items.length.toLocaleString()+' matching routes</p>'+
      (items.length?'<div class="table-wrap"><table><thead><tr><th>Route</th><th>Flights</th><th></th></tr></thead><tbody>'+slice.map(([k,list])=>'<tr><td><b>'+escapeHTML(k)+'</b><br>'+escapeHTML(list[0].origin)+' to '+escapeHTML(list[0].destination)+'</td><td>'+list.length+'</td><td><button class="linkbtn" data-flight="'+list[0].id+'" aria-label="Explore '+escapeHTML(k)+'">Explore</button></td></tr>').join('')+'</tbody></table></div><div class="pagination"><button class="linkbtn" id="prev">Previous</button><span>Page '+(routePage+1)+' of '+Math.ceil(items.length/8)+'</span><button class="linkbtn" id="next">Next</button></div>':'<p>Try another city or airport code.</p>');
    if (!items.length) return;
    $("prev").disabled=routePage===0; $("next").disabled=(routePage+1)*8>=items.length;
    $("prev").onclick=()=>{routePage--;display();}; $("next").onclick=()=>{routePage++;display();};
    document.querySelectorAll("[data-flight]").forEach(btn=>btn.onclick=()=>{current=flights[Number(btn.dataset.flight)];location.hash="prediction";});
  }
  $("search").oninput=()=>{routeFilter=$("search").value;routePage=0;display();};
  display();
}
function results() {
  layout('<h1>Compare Results</h1><label for="metric">Validation metric</label><select id="metric"><option value="pr_auc_mean">Average precision</option><option value="roc_auc_mean">ROC-AUC</option></select><p class="note">Four chronological folds and three seeds per checkpoint.</p>',
    panel("Model comparison","200,000-flight G1 validation",'<div class="body" id="subset"></div>') +
    panel("Feature comparison","Full-development LightGBM ablation",'<div class="body" id="scores"></div>'));
  const labels=["Schedule and route","+ Origin congestion","+ Surface weather","+ Inbound aircraft"];
  function update() {
    const column = metric === "pr_auc_mean" ? 2 : 1;
    const body = SUBSET_RESULTS.map(row =>
      "<tr><td>"+escapeHTML(row[0])+"</td><td>"+escapeHTML(row[column])+"</td><td>"+escapeHTML(row[column+2])+"</td></tr>"
    ).join("");
    $("subset").innerHTML = "<h2>"+escapeHTML($("metric").selectedOptions[0].textContent)+"</h2><p>Mean ± standard deviation</p><div class='table-wrap'><table><thead><tr><th>Model</th><th>6 hours</th><th>1 hour</th></tr></thead><tbody>"+body+"</tbody></table></div>";
    const rows=labels.map((name,i)=>{
      const value=cp=>EXPLORER_DATA.ablation.find(r=>r.group==="G"+i && r.checkpoint_minutes===cp)[metric].toFixed(4);
      return '<tr><td><b>G'+i+'</b> · '+name+(i>=2?' <small>(exploratory)</small>':'')+'</td><td>'+value(360)+'</td><td>'+value(60)+'</td></tr>';
    });
    $("scores").innerHTML='<h2>'+escapeHTML($("metric").selectedOptions[0].textContent)+'</h2><div class="table-wrap"><table><thead><tr><th>Feature set</th><th>6 hours</th><th>1 hour</th></tr></thead><tbody>'+rows.join('')+'</tbody></table></div><p class="caption">At one hour, adding recent origin congestion raised average precision from 0.3663 to 0.4151. Adding weather raised it to 0.4172.</p>';
  }
  $("metric").value=metric;$("metric").onchange=()=>{metric=$("metric").value;update();};update();
}
function about() {
  layout('<h1>About the Project</h1><p class="note">Flight departure delay prediction at two pre-departure checkpoints.</p>',
    panel("Flight Delay Predictor","Student Project",'<div class="body"><h2>What does the percentage mean?</h2><p>The estimated chance that a flight will leave its origin gate at least 15 minutes after schedule. The two checkpoints use information available six hours and one hour before scheduled departure.</p><h2>Where do the flights come from?</h2><p>The explorer replays 8,000 paired validation flights from the 2025 development pilot. Each percentage comes from saved LightGBM predictions at seed 42. “Reveal outcome” shows the recorded departure delay. Dates and scheduled times use UTC.</p><h2>What goes into a prediction?</h2><p>The pilot model uses schedule, route, calendar, and recent origin congestion. The Results page compares feature groups in a separate full-development experiment. Weather and inbound aircraft data are exploratory because their availability at the checkpoint is uncertain.</p><p><a href="../output/pdf/report.pdf">Read the project report</a> · <a href="https://github.com/oki-dokii/Flight-Departure-Delay-Prediction-">GitHub repository</a></p></div>'));
}
const pages={prediction,routes:routeList,results,about};
function render() {
  const hash=location.hash.slice(1);
  const page=Object.hasOwn(pages,hash)?hash:"prediction";
  document.querySelectorAll("nav a").forEach(a=>{if(a.hash==="#"+page)a.setAttribute("aria-current","page");else a.removeAttribute("aria-current");});
  pages[page]();document.title=document.querySelector("h1").textContent+" · Flight Delay Predictor";
}
window.addEventListener("hashchange",render);render();
