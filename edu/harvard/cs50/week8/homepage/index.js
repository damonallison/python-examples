function greet() {
    alert("hello, " + document.querySelector("#query").value);
}

// wait for the entire DOM to load before attaching events
document.addEventListener("DOMContentLoaded", function () {
    document.querySelector("#fortune-button").addEventListener('click', function (e) {
        const currentSecond = new Date().getSeconds();
        var fortune = "In Jesus we love.";
        if (currentSecond % 2 == 0) {
            fortune = "In God we trust.";
        }
        document.querySelector("#fortune").innerHTML = fortune;
        e.preventDefault();
    });
});
