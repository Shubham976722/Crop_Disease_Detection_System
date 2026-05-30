// Navbar Scroll Effect
window.addEventListener("scroll", function() {
    let nav = document.getElementById("mainNav");
    if (window.scrollY > 50) {
        nav.classList.add("scrolled");
    } else {
        nav.classList.remove("scrolled");
    }
});

// Image Preview
function previewImage(input) {
    let preview = document.getElementById("previewImg");
    let file = input.files[0];
    let reader = new FileReader();

    reader.onload = function(e) {
        preview.src = e.target.result;
        preview.style.display = "block";
    }

    reader.readAsDataURL(file);
}

// Loading Spinner
function showSpinner() {
    document.getElementById("spinner").style.display = "flex";
}

// Dark / Light Toggle
function toggleTheme() {
    let html = document.documentElement;
    let current = html.getAttribute("data-bs-theme");
    html.setAttribute("data-bs-theme", current === "dark" ? "light" : "dark");
}