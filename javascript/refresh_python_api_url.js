function bayesian_merger_onload() {
    api_url_textarea = gradioApp().querySelector("#bayesian_merger_api_url textarea");

    // on some browsers, the load event is triggered too soon
    // retry to pass the url to gradio every 100ms
    if (api_url_textarea === null) {
        setTimeout(bayesian_merger_onload, 100);
        return;
    }

    refresh_python_api_url(api_url_textarea)
}

window.addEventListener("load", bayesian_merger_onload);

function refresh_python_api_url(api_url_textarea) {
    api_url = window.location.href.split('?')[0].slice(0, -1);

    api_url_textarea.value = api_url;
    event = new Event("input");

    // for some reason `event.target` is null on load
    // force `event.target` to a dummy value that gradio javascript code will happily work with
    dummy_target = { style: { height: "" } };
    Object.defineProperty(event, "target", { writable: false, value: dummy_target });

    api_url_textarea.dispatchEvent(event);
}
