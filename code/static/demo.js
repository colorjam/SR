window.onload = function() {
    var submit = document.getElementsByClassName('form-submit')[0]
    var loader = document.getElementsByClassName('loader')[0]
    console.log(loader)
    submit.onclick = () => loader.style.display = 'block'


}