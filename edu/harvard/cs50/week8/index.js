function greet() {
    alert("hello, " + document.querySelector("#query").value);
}

// wait for the entire DOM to load before attaching events
document.addEventListener("DOMContentLoaded", function () {
    document.querySelector("#local-button").addEventListener('click', function (e) {
        alert("hello, " + document.querySelector("#local-query").value);
        e.preventDefault();
    });
});
