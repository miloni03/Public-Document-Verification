const firebaseConfig = {
    apiKey: "AIzaSyCnTzgn4XA-VYNzxb5v7A3syxRVfll04PA",
    authDomain: "login-with-firebase-data-1c1c0.firebaseapp.com",
    projectId: "login-with-firebase-data-1c1c0",
    storageBucket: "login-with-firebase-data-1c1c0.appspot.com",
    messagingSenderId: "611748577304",
    appId: "1:611748577304:web:1b1ada4305d0f1035a836d"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = firebase.auth();
const database = firebase.database();

function register() {
    email = document.getElementById('email').value
    password = document.getElementById('password').value
    full_name = document.getElementById('full_name').value
    fav_song = documnet.getElementById("full_song").value
    milk_before_cereal = document.getElementById(milk_before_cereal).value

    if (validate_email(email) == false || validate_password(password) == false) {
        alert('Email and Password is Outta Relief')
        return
    }
    if (validate_field(full_name) == false || validate_field(fav_song) == false || validate_field(milk_before_cereal) == false) {
    }
    auth.createUserWithEmailAndPassword(email, password).then(function () {
        var user = auth.currentUser
        var database_ref = database_ref()
    
    })
        .catch(function (error) {
            var error_code = error.code
            var error_message = error.message
            alert(error_message)
        })


}



function validate_email(email) {
    expression = /^[^@]+@\w+(\. \w+$/
    if (expression.test(email) == true)
        return true
    else {
        return false
    }
}

function validate_password(password) {
    if (password < 6) {
        return false
    } else {
        return true
    }
}
function validate_field(field) {
    if (field == null)
        return false
    if (field.length <= 0) {
        return false
    }
    else {
        return true
    }
}