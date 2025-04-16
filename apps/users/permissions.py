from rest_framework import permissions

class IsClinician(permissions.BasePermission):
    """ Allows access only to users with CLINICIAN role. """
    message = "Only clinicians are allowed to perform this action."

    def has_permission(self, request, view):
        return bool(request.user and request.user.is_authenticated and request.user.role == 'CLINICIAN')

class IsPatient(permissions.BasePermission):
    """ Allows access only to users with PATIENT role. """
    message = "Only patients are allowed to perform this action."

    def has_permission(self, request, view):
         return bool(request.user and request.user.is_authenticated and request.user.role == 'PATIENT')

class IsPatientOwner(permissions.BasePermission):
    """ Allows access only to the patient who owns the profile/medication. """
    message = "You do not have permission to access this patient data."

    def has_object_permission(self, request, view, obj):
        # Read permissions are allowed to any request,
        # so we'll always allow GET, HEAD or OPTIONS requests.
        if request.method in permissions.SAFE_METHODS:
             # Allow access if user is admin or the patient themselves
            return request.user.is_staff or (hasattr(obj, 'user') and obj.user == request.user) or (hasattr(obj, 'patient_profile') and obj.patient_profile.user == request.user)

        # Write permissions are only allowed to the owner of the data.
        # Check if the object has a 'user' attribute (like PatientProfile)
        if hasattr(obj, 'user'):
            return obj.user == request.user
        # Check if the object has a 'patient_profile' attribute (like Medication)
        elif hasattr(obj, 'patient_profile'):
             return obj.patient_profile.user == request.user
        return False # Default deny if ownership cannot be determined