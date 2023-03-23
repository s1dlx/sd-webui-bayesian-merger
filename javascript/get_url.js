function bayesian_merger_load() {
    api_url_textarea = gradioApp().querySelector("#bayesian_merger_api_url textarea");

    // on some browsers, the load event is triggered too soon
    // retry to pass the url to gradio every 100ms
    if (api_url_textarea === null) {
        setTimeout(bayesian_merger_load, 100);
        return;
    }

    api_url_textarea.value = window.location.href;
    event = new Event("input", {
        bubbles: false,
    });

    // for some reason the target of the event is null on Firefox
    // force the target to a dummy value gradio will happily work with
    dummy_target = { style: { height: "" } };
    Object.defineProperty(event, "target", { writable: false, value: dummy_target });

    api_url_textarea.dispatchEvent(event);
}

window.addEventListener("load", bayesian_merger_load);
