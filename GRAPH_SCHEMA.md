# Graph Schema (Phase 1)

## Nodes
- Person
- SupporterType
- Tag
- InvolvementArea
- Skill
- Address
- Campaign
- Nation
- Broadcaster
- PaymentProcessor
- Goal
- Path
- Activity
- Event
- Contribution

## Relationships
- (Person)-[:LIVES_AT]->(Address)
- (Person)-[:CLASSIFIED_AS]->(SupporterType)
- (Person)-[:HAS_TAG]->(Tag)
- (Person)-[:INTERESTED_IN]->(InvolvementArea)
- (Person)-[:CAN_CONTRIBUTE_WITH]->(Skill)
- (Person)-[:REFERRED_BY]->(Person)
- (Person)-[:SUPPORTS]->(Campaign)
- (Person)-[:VOLUNTEERS_FOR]->(Campaign)
- (Person)-[:DONATED_TO]->(Campaign)
- (Person)-[:IN_PATH]->(Path)
- (Person)-[:HAS_ACTIVITY]->(Activity)
- (Person)-[:REGISTERED_FOR]->(Event)
- (Person)-[:MADE_CONTRIBUTION]->(Contribution)
- (Nation)-[:HAS_BROADCASTER]->(Broadcaster)
- (Nation)-[:USES_PAYMENT_PROCESSOR]->(PaymentProcessor)
- (Nation)-[:HAS_GOAL]->(Goal)
- (Nation)-[:HAS_PATH]->(Path)
- (Goal)-[:HAS_PATH]->(Path)
- (Contribution)-[:PROCESSED_BY]->(PaymentProcessor)
- (Contribution)-[:ATTRIBUTED_TO]->(Campaign)
- (Activity)-[:RELATED_TO]->(Campaign)

## Core properties (selected)
- Person: personId, email, firstName, lastName, phone, gender, age, about, timeAvailability,
  agreesWithManifesto, interestedInMembership, facebookGroupMember, lat, lon, donationTotal, createdAt
- Address: fullAddress, latitude, longitude
- Campaign: name
- Nation: slug, name, website, timezone, contactName, contactEmail, phone, address,
  billingEmail, billingAddress, ownerEmail
- Broadcaster: broadcasterKey, name, role, email, twitter, facebook, phone
- PaymentProcessor: provider, accountId, currency, liveMode
- Goal: goalKey, name, description, unit, targetValue, currentValue
- Path: pathKey, name, description, steps
- Activity: activityId, type, subject, status, activityDate, details
- Event: eventKey, eventId, name, startDate, endDate, location, status, capacity, notes
- Contribution: contributionId, amount, currency, receiveDate, status, source, note

## Constraints and indexes
- Person.email unique
- District.name unique (optional)
- Address geo index on (latitude, longitude)
- Event.eventKey unique (recommended)

## Notes
- personId uses randomUUID() on create.
- Nation nodes represent separate "nations" (tenants) via slug.
- Tags, skills, and involvement areas are normalized nodes for segmentation.
